#!/usr/bin/env python3

import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

# Data processing
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Dask for distributed processing
import dask
from dask.distributed import Client, as_completed
from dask import delayed
import dask.bag as db

# Geospatial libraries
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
from shapely.ops import unary_union
import rasterio
from rasterio.crs import CRS

# Custom imports
from large_scale_functions import load_inference_model, convert_to_polygons, assess_quality
from dask_model_loader import UNetWithHeight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting automated pipeline...")
logger.info(f"Platform: {sys.platform}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Available CPU cores: {mp.cpu_count()}")
logger.info(f"Multiprocessing start method: {mp.get_start_method()}")

class AutomatedVectorPipeline:
    """
    Automated pipeline for processing
    
    """
    
    def __init__(self, 
                 outputs_dir: str = 'height_analysis_output/vector_map_exports',
                 output_base: str = 'vector_processing_output',
                 pixel_size: float = 0.5,
                 crs: str = 'EPSG:4326',
                 max_workers: int = None):
        """
        Initialize the automated pipeline
        
        """
        self.outputs_dir = Path(outputs_dir)
        self.output_base = Path.cwd() / output_base
        self.pixel_size = pixel_size
        self.crs = crs
        self.max_workers = min(max_workers or mp.cpu_count(), 2) 
        
        if sys.platform == 'win32':
            logger.info("Windows detected - using conservative Dask settings")
            logger.info("Using max 2 workers to prevent Windows deadlocks")
            logger.info("Process-based workers enabled for Windows stability")
        
        # Create output directories
        self.output_base.mkdir(exist_ok=True, parents=True)
        self.vectors_dir = self.output_base / 'individual_patches'
        self.vectors_dir.mkdir(exist_ok=True)
        self.unified_dir = self.output_base / 'unified_database'
        self.unified_dir.mkdir(exist_ok=True)
        self.qgis_dir = self.output_base / 'qgis_projects'
        self.qgis_dir.mkdir(exist_ok=True)
        
        # Initialize statistics
        self.processing_stats = {
            'total_patches': 0,
            'successful_patches': 0,
            'failed_patches': 0,
            'total_buildings': 0,
            'total_area_m2': 0.0,
            'processing_time': 0.0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info(f"Initialized AutomatedVectorPipeline")
        logger.info(f"Output directory: {self.output_base}")
        logger.info(f"Max workers: {self.max_workers}")
    
    def discover_patches(self) -> List[str]:
        """
        Discover all available patches from height model outputs
        
        """
        logger.info("Discovering available patches...")
        
        # Find all segmentation files
        segmentation_files = list(self.outputs_dir.glob('*_segmentation.png'))
        height_files = list(self.outputs_dir.glob('*_height.png'))
        metadata_files = list(self.outputs_dir.glob('*_metadata.json'))
        
        # Extract patch numbers
        patch_numbers = []
        for seg_file in segmentation_files:
            patch_name = seg_file.stem.replace('_segmentation', '')
            
            # Verify all required files exist
            height_file = self.outputs_dir / f"{patch_name}_height.png"
            metadata_file = self.outputs_dir / f"{patch_name}_metadata.json"
            
            if height_file.exists() and metadata_file.exists():
                patch_numbers.append(patch_name)
        
        patch_numbers = sorted(patch_numbers)
        
        logger.info(f"Found {len(patch_numbers)} complete patches")
        logger.info(f"Segmentation files: {len(segmentation_files)}")
        logger.info(f"Height files: {len(height_files)}")
        logger.info(f"Metadata files: {len(metadata_files)}")
        
        if patch_numbers:
            logger.info(f"Patch range: {patch_numbers[0]} to {patch_numbers[-1]}")
        
        return patch_numbers
    
    def process_single_patch(self, patch_name: str) -> Dict:
        """
        Process a single patch and return results
        
        """
        try:
            # Define file paths
            seg_path = self.outputs_dir / f"{patch_name}_segmentation.png"
            height_path = self.outputs_dir / f"{patch_name}_height.png"
            metadata_path = self.outputs_dir / f"{patch_name}_metadata.json"
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                patch_metadata = json.load(f)
            
            # Load segmentation mask
            segmentation_img = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
            if segmentation_img is None:
                raise ValueError(f"Could not load segmentation: {seg_path}")
            
            segmentation = segmentation_img.astype(np.float32) / 255.0
            
            # Load height map
            height_img = cv2.imread(str(height_path), cv2.IMREAD_GRAYSCALE)
            if height_img is None:
                raise ValueError(f"Could not load height map: {height_path}")
            
            height_scale = patch_metadata.get('height_scale_factor', 50.0)
            height_map = height_img.astype(np.float32) * height_scale / 255.0
            
            # Convert to polygons using the VectorConverter logic
            polygons = self.mask_to_polygons_advanced(
                mask=segmentation,
                height_map=height_map,
                threshold=0.5,
                min_area=50,
                simplify_tolerance=1.0
            )
            
            if not polygons:
                return {
                    'patch_name': patch_name,
                    'success': False,
                    'error': 'No polygons generated',
                    'buildings_count': 0,
                    'total_area_m2': 0.0
                }
            
            # Apply topology correction
            corrected_polygons = self.topology_correction(polygons)
            
            # Create GeoDataFrame
            gdf = self.create_geodataframe(corrected_polygons, patch_name)
            
            if gdf.empty:
                return {
                    'patch_name': patch_name,
                    'success': False,
                    'error': 'Empty GeoDataFrame',
                    'buildings_count': 0,
                    'total_area_m2': 0.0
                }
            
            # Export individual patch results
            patch_output_dir = self.vectors_dir / f"{patch_name}_vectors"
            self.export_patch_vectors(gdf, patch_output_dir, patch_name, patch_metadata)
            
            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(gdf, patch_metadata)
            
            # Save quality assessment
            quality_file = patch_output_dir / 'quality_assessment.json'
            with open(quality_file, 'w', encoding='utf-8') as f:
                json.dump(quality_metrics, f, indent=2)
            
            return {
                'patch_name': patch_name,
                'success': True,
                'buildings_count': len(gdf),
                'total_area_m2': float(gdf['area_m2'].sum()),
                'quality_score': quality_metrics.get('overall_quality', 0),
                'gdf': gdf,
                'output_dir': patch_output_dir
            }
            
        except Exception as e:
            logger.error(f"Error processing patch {patch_name}: {str(e)}")
            return {
                'patch_name': patch_name,
                'success': False,
                'error': str(e),
                'buildings_count': 0,
                'total_area_m2': 0.0
            }
    
    def mask_to_polygons_advanced(self, mask, height_map=None, threshold=0.5, 
                                 min_area=50, simplify_tolerance=1.0):
        """
        Convert segmentation mask to polygons with advanced processing
    
        """
        # Convert to binary mask
        binary_mask = (mask > threshold).astype(np.uint8)
        
        # Apply morphological operations for cleanup
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                # Simplify polygon
                epsilon = simplify_tolerance * cv2.arcLength(contour, True) / 100
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(simplified) >= 3:
                    # Convert to shapely polygon
                    coords = simplified.squeeze().tolist()
                    if len(coords) >= 3:
                        try:
                            # Convert pixel coordinates to real-world coordinates
                            real_coords = [(x * self.pixel_size, y * self.pixel_size) for x, y in coords]
                            poly = Polygon(real_coords)
                            
                            # Validate and fix topology
                            if not poly.is_valid:
                                poly = make_valid(poly)
                            
                            if poly.is_valid and not poly.is_empty:
                                # Calculate properties
                                properties = {
                                    'building_id': f'building_{i:04d}',
                                    'area_m2': float(poly.area),
                                    'perimeter_m': float(poly.length),
                                    'confidence': float(np.mean(mask[binary_mask == 1]) if np.any(binary_mask) else 0.0)
                                }
                                
                                # Add height information if available
                                if height_map is not None:
                                    # Get height statistics for this polygon
                                    mask_poly = np.zeros_like(binary_mask)
                                    cv2.fillPoly(mask_poly, [simplified], 1)
                                    height_values = height_map[mask_poly == 1]
                                    
                                    if len(height_values) > 0:
                                        properties.update({
                                            'height_mean': float(np.mean(height_values)),
                                            'height_max': float(np.max(height_values)),
                                            'height_std': float(np.std(height_values))
                                        })
                                
                                polygons.append({
                                    'geometry': poly,
                                    'properties': properties
                                })
                                
                        except Exception as e:
                            logger.warning(f"Error processing polygon {i}: {e}")
                            continue
        
        return polygons
    
    def topology_correction(self, polygons, buffer_distance=0.1):
        """
        Apply topology correction to polygons
        """
        corrected_polygons = []
        
        for poly_dict in polygons:
            try:
                poly = poly_dict['geometry']
                
                # Apply small buffer to fix topology issues
                buffered = poly.buffer(buffer_distance).buffer(-buffer_distance)
                
                if buffered.is_valid and not buffered.is_empty:
                    # Handle multipolygon results
                    if buffered.geom_type == 'MultiPolygon':
                        # Take the largest polygon
                        largest_poly = max(buffered.geoms, key=lambda x: x.area)
                        if largest_poly.area > 10:  # Minimum area threshold
                            poly_dict['geometry'] = largest_poly
                            corrected_polygons.append(poly_dict)
                    else:
                        poly_dict['geometry'] = buffered
                        corrected_polygons.append(poly_dict)
                        
            except Exception as e:
                logger.warning(f"Error in topology correction: {e}")
                continue
        
        return corrected_polygons
    
    def create_geodataframe(self, polygons, patch_name):
        """
        Create GeoDataFrame from polygons
        """
        if not polygons:
            return gpd.GeoDataFrame()
        
        # Extract geometries and properties
        geometries = [p['geometry'] for p in polygons]
        properties = [p['properties'] for p in polygons]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=self.crs)
        
        # Add patch information
        gdf['patch_name'] = patch_name
        gdf['processing_date'] = datetime.now().isoformat()
        
        # Add additional calculated fields
        gdf['shape_length'] = gdf.geometry.length
        gdf['shape_area'] = gdf.geometry.area
        gdf['compactness'] = (4 * np.pi * gdf['shape_area']) / (gdf['shape_length'] ** 2)
        
        return gdf
    
    def export_patch_vectors(self, gdf, output_dir, patch_name, patch_metadata):
        """
        Export vector data for a single patch
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Export to multiple formats
        base_name = f"{patch_name}_buildings"
        
        # GeoJSON
        geojson_path = output_dir / f"{base_name}.geojson"
        gdf.to_file(geojson_path, driver='GeoJSON')
        
        # Shapefile
        shapefile_path = output_dir / f"{base_name}.shp"
        gdf.to_file(shapefile_path, driver='ESRI Shapefile')
        
        # GeoPackage
        gpkg_path = output_dir / f"{base_name}.gpkg"
        gdf.to_file(gpkg_path, driver='GPKG')
        
        # CSV summary
        csv_path = output_dir / f"{base_name}_summary.csv"
        gdf.to_csv(csv_path, index=False)
        
        # Processing metadata
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'patch_name': patch_name,
            'original_metadata': patch_metadata,
            'vector_processing': {
                'total_buildings': len(gdf),
                'total_area_m2': float(gdf['area_m2'].sum()),
                'average_area_m2': float(gdf['area_m2'].mean()),
                'pixel_size_m': self.pixel_size,
                'coordinate_system': self.crs,
                'has_height_data': 'height_mean' in gdf.columns
            }
        }
        
        metadata_path = output_dir / 'processing_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def calculate_quality_metrics(self, gdf, patch_metadata):
        """
        Calculate quality metrics for a patch
        """
        quality_metrics = {
            'geometry_validation': {
                'total_polygons': len(gdf),
                'valid_polygons': sum(gdf.geometry.is_valid),
                'validation_rate': sum(gdf.geometry.is_valid) / len(gdf) * 100
            },
            'geometric_properties': {
                'total_area_m2': float(gdf['area_m2'].sum()),
                'average_area_m2': float(gdf['area_m2'].mean()),
                'median_area_m2': float(gdf['area_m2'].median())
            },
            'confidence_metrics': {
                'average_confidence': float(gdf['confidence'].mean()),
                'min_confidence': float(gdf['confidence'].min()),
                'max_confidence': float(gdf['confidence'].max())
            }
        }
        
        # Add height metrics if available
        if 'height_mean' in gdf.columns:
            quality_metrics['height_metrics'] = {
                'average_height': float(gdf['height_mean'].mean()),
                'max_height': float(gdf['height_mean'].max()),
                'min_height': float(gdf['height_mean'].min())
            }
        
        # Calculate overall quality score
        validation_score = quality_metrics['geometry_validation']['validation_rate']
        confidence_score = quality_metrics['confidence_metrics']['average_confidence'] * 100
        overall_quality = (validation_score + confidence_score) / 2
        
        quality_metrics['overall_quality'] = overall_quality
        
        return quality_metrics
    
    def run_batch_processing(self, patch_subset=None, max_patches=None):
        """
        Run batch processing on all available patches using Dask
        
        """
        logger.info("Starting batch processing with Dask...")
        self.processing_stats['start_time'] = datetime.now()
        
        # Discover patches
        if patch_subset:
            patch_numbers = patch_subset
        else:
            patch_numbers = self.discover_patches()
        
        if max_patches:
            patch_numbers = patch_numbers[:max_patches]
        
        self.processing_stats['total_patches'] = len(patch_numbers)
        
        logger.info(f"Processing {len(patch_numbers)} patches with Dask (Large Scale: ~18,000 patches)")
        
        # Initialize Dask client
        with Client(
            n_workers=min(2, self.max_workers), 
            threads_per_worker=1,  
            memory_limit='2GB',  
            processes=True, 
            silence_logs=False
        ) as client:
            logger.info(f"Dask client initialized: {client.dashboard_link}")
            
            # Process patches in smaller batches for memory management
            batch_size = min(10, len(patch_numbers)) 
            successful_results = []
            failed_results = []
            
            for i in range(0, len(patch_numbers), batch_size):
                batch_patches = patch_numbers[i:i + batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(patch_numbers) + batch_size - 1)//batch_size} of 18,000 patches")
                logger.info(f"Batch size: {len(batch_patches)} patches")
                
                # Add progress checkpoint every 50 batches for stability
                if (i//batch_size + 1) % 50 == 0:
                    logger.info(f"Checkpoint: {i//batch_size + 1} batches completed")
                    logger.info(f"Progress: {len(successful_results) + len(failed_results)}/{len(patch_numbers)} patches processed")
                    # Force garbage collection on Windows for memory management
                    import gc
                    gc.collect()
                
                # Create delayed tasks for this batch
                delayed_tasks = [
                    delayed(self.process_single_patch)(patch_name) 
                    for patch_name in batch_patches
                ]
                
                try:
                    # Process batch
                    batch_futures = client.compute(delayed_tasks)
                    batch_results = client.gather(batch_futures)
                    
                    # Process results
                    for result, patch_name in zip(batch_results, batch_patches):
                        if result and result.get('success', False):
                            successful_results.append(result)
                            self.processing_stats['successful_patches'] += 1
                            self.processing_stats['total_buildings'] += result['buildings_count']
                            self.processing_stats['total_area_m2'] += result['total_area_m2']
                        else:
                            failed_results.append({
                                'patch_name': patch_name,
                                'success': False,
                                'error': result.get('error', 'Unknown error') if result else 'No result returned',
                                'buildings_count': 0,
                                'total_area_m2': 0.0
                            })
                            self.processing_stats['failed_patches'] += 1
                    
                    # Log batch progress with enhanced reporting for large scale
                    total_processed = len(successful_results) + len(failed_results)
                    progress = (total_processed / len(patch_numbers)) * 100
                    remaining = len(patch_numbers) - total_processed
                    
                    logger.info(f"Batch {i//batch_size + 1} complete. Progress: {total_processed:,}/{len(patch_numbers):,} ({progress:.1f}%)")
                    logger.info(f"Remaining patches: {remaining:,}")
                    
                    # Estimate completion time
                    if total_processed > 0:
                        elapsed = (datetime.now() - self.processing_stats['start_time']).total_seconds()
                        rate = total_processed / elapsed
                        eta_seconds = remaining / rate if rate > 0 else 0
                        eta_hours = eta_seconds / 3600
                        logger.info(f"Estimated completion: {eta_hours:.1f} hours")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    for patch_name in batch_patches:
                        failed_results.append({
                            'patch_name': patch_name,
                            'success': False,
                            'error': f'Batch processing error: {str(e)}',
                            'buildings_count': 0,
                            'total_area_m2': 0.0
                        })
                        self.processing_stats['failed_patches'] += 1
        
        self.processing_stats['end_time'] = datetime.now()
        self.processing_stats['processing_time'] = (
            self.processing_stats['end_time'] - self.processing_stats['start_time']
        ).total_seconds()
        
        logger.info("Batch processing completed!")
        self.log_processing_summary(successful_results, failed_results)
        
        return successful_results, failed_results
    
    def log_processing_summary(self, successful_results, failed_results):
        """
        Log processing summary statistics
        """
        logger.info("=== BATCH PROCESSING SUMMARY ===")
        logger.info(f"Total patches: {self.processing_stats['total_patches']:,}")
        logger.info(f"Successful: {self.processing_stats['successful_patches']:,}")
        logger.info(f"Failed: {self.processing_stats['failed_patches']:,}")
        logger.info(f"Success rate: {self.processing_stats['successful_patches']/self.processing_stats['total_patches']*100:.1f}%")
        logger.info(f"Total buildings extracted: {self.processing_stats['total_buildings']:,}")
        logger.info(f"Total building area: {self.processing_stats['total_area_m2']:,.2f} m²")
        logger.info(f"Processing time: {self.processing_stats['processing_time']:.2f} seconds ({self.processing_stats['processing_time']/3600:.1f} hours)")
        logger.info(f"Average time per patch: {self.processing_stats['processing_time']/self.processing_stats['total_patches']:.2f} seconds")
        logger.info(f"Processing rate: {self.processing_stats['total_patches']/(self.processing_stats['processing_time']/3600):.1f} patches/hour")
        
        # Save summary to file
        summary_file = self.output_base / 'processing_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'processing_stats': {
                    **self.processing_stats,
                    'start_time': self.processing_stats['start_time'].isoformat(),
                    'end_time': self.processing_stats['end_time'].isoformat()
                },
                'successful_patches': [r['patch_name'] for r in successful_results],
                'failed_patches': [r['patch_name'] for r in failed_results]
            }, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_file}")
    
    def create_unified_database(self, successful_results):
        """
        Create unified spatial database from all processed patches using Dask
        """
        logger.info("Creating unified spatial database with Dask...")
        
        if not successful_results:
            logger.warning("No successful results to create unified database")
            return
        
        # For memory efficiency, load GeoDataFrames from files instead of keeping in memory
        patch_dirs = [result['output_dir'] for result in successful_results]
        
        logger.info(f"Loading {len(patch_dirs)} patch results...")
        
        # Use Dask to load and process GeoDataFrames efficiently
        @delayed
        def load_patch_gdf(patch_dir):
            """Load a single patch GeoDataFrame from file"""
            patch_name = Path(patch_dir).name.replace('_vectors', '')
            gpkg_path = Path(patch_dir) / f"{patch_name}_buildings.gpkg"
            
            if gpkg_path.exists():
                return gpd.read_file(gpkg_path)
            else:
                return gpd.GeoDataFrame()
        
        # Create delayed tasks for loading all patches
        delayed_gdfs = [load_patch_gdf(patch_dir) for patch_dir in patch_dirs]
        
        # Initialize Dask client for database creation 
        with Client(
            n_workers=1,  
            threads_per_worker=1,  
            memory_limit='4GB',  
            processes=True, 
            silence_logs=False
        ) as client:
            logger.info(f"Dask client for database creation: {client.dashboard_link}")
            
            # Process GeoDataFrames in smaller chunks for memory management
            chunk_size = 50 
            all_gdfs = []
            
            for i in range(0, len(delayed_gdfs), chunk_size):
                chunk_delayed = delayed_gdfs[i:i + chunk_size]
                logger.info(f"Loading chunk {i//chunk_size + 1}/{(len(delayed_gdfs) + chunk_size - 1)//chunk_size}")
                
                chunk_futures = client.compute(chunk_delayed)
                chunk_gdfs = client.gather(chunk_futures)
                valid_chunk_gdfs = [gdf for gdf in chunk_gdfs if not gdf.empty]
                all_gdfs.extend(valid_chunk_gdfs)
                
                logger.info(f"Loaded {len(valid_chunk_gdfs)} valid GeoDataFrames from chunk")
            
            if not all_gdfs:
                logger.warning("No valid GeoDataFrames found")
                return
            
            # Combine all GeoDataFrames
            logger.info(f"Combining {len(all_gdfs)} GeoDataFrames...")
            
            # Use Dask for memory-efficient concatenation
            unified_gdf = gpd.pd.concat(all_gdfs, ignore_index=True)
            
            # Add global building IDs
            unified_gdf['global_building_id'] = range(1, len(unified_gdf) + 1)
            
            # Export unified database
            unified_base = self.unified_dir / 'unified_buildings'
            
            logger.info("Exporting unified database...")
            
            # Export in chunks to manage memory
            chunk_size = 10000
            
            if len(unified_gdf) > chunk_size:
                logger.info(f"Large dataset ({len(unified_gdf)} buildings), exporting in chunks...")
                
                # Export GeoPackage (best for large datasets)
                unified_gdf.to_file(f"{unified_base}.gpkg", driver='GPKG')
                
                # Export CSV in chunks
                for i in range(0, len(unified_gdf), chunk_size):
                    chunk = unified_gdf.iloc[i:i+chunk_size]
                    chunk_file = self.unified_dir / f'unified_buildings_chunk_{i//chunk_size + 1}.csv'
                    chunk.to_csv(chunk_file, index=False)
                
                # Also create a complete CSV
                unified_gdf.to_csv(f"{unified_base}.csv", index=False)
                
            else:
                # Standard export for smaller datasets
                unified_gdf.to_file(f"{unified_base}.geojson", driver='GeoJSON')
                unified_gdf.to_file(f"{unified_base}.shp", driver='ESRI Shapefile')
                unified_gdf.to_file(f"{unified_base}.gpkg", driver='GPKG')
                unified_gdf.to_csv(f"{unified_base}.csv", index=False)
            
            logger.info(f"Unified database created with {len(unified_gdf):,} buildings")
            logger.info(f"Total area: {unified_gdf['area_m2'].sum():,.2f} square meters")
            
            # Calculate and log statistics
            stats = {
                'total_buildings': len(unified_gdf),
                'total_area_m2': float(unified_gdf['area_m2'].sum()),
                'average_area_m2': float(unified_gdf['area_m2'].mean()),
                'patches_processed': len(all_gdfs),
                'has_height_data': 'height_mean' in unified_gdf.columns
            }
            
            if 'height_mean' in unified_gdf.columns:
                stats['height_statistics'] = {
                    'average_height': float(unified_gdf['height_mean'].mean()),
                    'max_height': float(unified_gdf['height_mean'].max()),
                    'min_height': float(unified_gdf['height_mean'].min())
                }
            
            # Save database statistics
            stats_file = self.unified_dir / 'database_statistics.json'
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Database statistics saved to: {stats_file}")
            
            return unified_gdf
    
    def generate_qgis_batch_script(self, successful_results):
        """
        Generate QGIS batch processing script for verification
        """
        logger.info("Generating QGIS batch processing script for large-scale verification...")
        
    
        total_patches = len(successful_results)
        
        # Different sampling strategies for comprehensive verification
        first_500 = successful_results[:500]  # First 500 patches
        random_sample = successful_results[::max(1, total_patches//1000)]  # Every nth patch for ~1000 samples
        
        qgis_script = f"""
# QGIS Batch Processing Script for Large-Scale Vector Verification
# Generated: {datetime.now().isoformat()}
# Total successful patches: {len(successful_results):,} out of 18,000+

import processing
from qgis.core import QgsVectorLayer, QgsProject, QgsLayerTreeGroup

# Load unified database
unified_layer = QgsVectorLayer(
    r"{self.unified_dir}/unified_buildings.gpkg",
    "All_Buildings_Unified",
    "ogr"
)
QgsProject.instance().addMapLayer(unified_layer)

print(f"📊 Large-scale verification: {len(successful_results):,} successful patches loaded")

# Create layer groups for better organization
root = QgsProject.instance().layerTreeRoot()
unified_group = root.addGroup("📊 Unified Database")
samples_group = root.addGroup("🔍 Sample Patches")

# Move unified layer to its group
unified_group.addLayer(unified_layer)

# Load first 500 patches for systematic verification
first_500_patches = [
"""
        
        # Add first 500 patches
        for result in first_500:
            patch_name = result['patch_name']
            vector_path = self.vectors_dir / f"{patch_name}_vectors" / f"{patch_name}_buildings.gpkg"
            qgis_script += f'    r"{vector_path}",\n'
        
        qgis_script += f"""
]

# Load random sample patches for distribution analysis
random_sample_patches = [
"""
        
        # Add random sample patches
        for result in random_sample:
            patch_name = result['patch_name']
            vector_path = self.vectors_dir / f"{patch_name}_vectors" / f"{patch_name}_buildings.gpkg"
            qgis_script += f'    r"{vector_path}",\n'
        
        qgis_script += f"""
]

# Create subgroups for different sampling strategies
first_500_group = samples_group.addGroup("First 500 Patches")
random_sample_group = samples_group.addGroup("Random Sample ({len(random_sample)} patches)")

# Load first 500 patches
for i, patch_path in enumerate(first_500_patches):
    layer = QgsVectorLayer(patch_path, f"First_{{i+1:03d}}", "ogr")
    if layer.isValid():
        QgsProject.instance().addMapLayer(layer)
        first_500_group.addLayer(layer)

# Load random sample patches
for i, patch_path in enumerate(random_sample_patches):
    layer = QgsVectorLayer(patch_path, f"Random_{{i+1:03d}}", "ogr")
    if layer.isValid():
        QgsProject.instance().addMapLayer(layer)
        random_sample_group.addLayer(layer)

# Summary statistics
total_buildings = {self.processing_stats['total_buildings']:,}
total_area = {self.processing_stats['total_area_m2']:,.2f}
success_rate = {len(successful_results)/self.processing_stats['total_patches']*100:.1f}

print(f"✅ Verification Setup Complete:")
print(f"   📈 Total patches processed: {self.processing_stats['total_patches']:,}")
print(f"   🎯 Successful patches: {len(successful_results):,}")
print(f"   📊 Success rate: {{success_rate:.1f}}%")
print(f"   🏢 Total buildings: {{total_buildings:,}}")
print(f"   📐 Total area: {{total_area:,.2f}} square meters")
print(f"   🔍 Loaded {{len(first_500_patches)}} + {{len(random_sample_patches)}} patches for verification")

# Style by height if available
if 'height_mean' in [field.name() for field in unified_layer.fields()]:
    print("🏗️ Height data available - consider applying graduated symbology")
    
print("🎉 Large-scale verification ready!")
"""

        
        qgis_script_path = self.qgis_dir / 'batch_verification.py'
        with open(qgis_script_path, 'w', encoding='utf-8') as f:
            f.write(qgis_script)
        
        
        
        logger.info(f"QGIS script generated: {qgis_script_path}")
        

def main():
    """
    Main function to run the automated pipeline
    """
    print("Starting Automated Vector Processing Pipeline")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = AutomatedVectorPipeline(
        max_workers=2 
    )
    
    # Run batch processing on ALL available patches
    successful_results, failed_results = pipeline.run_batch_processing(
        # max_patches=100 
    )
    
    # Create unified database from successful results
    if successful_results:
        print(f"🔄 Creating unified database from {len(successful_results)} successful patches...")
        unified_gdf = pipeline.create_unified_database(successful_results)
        
        # Generate QGIS batch script
        pipeline.generate_qgis_batch_script(successful_results)
    
    print("\nAutomated pipeline completed!")
    print(f"Scale: Processed {len(successful_results):,} patches successfully")

if __name__ == "__main__":
    main()
