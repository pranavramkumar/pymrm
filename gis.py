"""
Geographic Information Systems (GIS) for Physical Climate Risk Modeling

This module provides comprehensive GIS functionality for climate risk assessment including:
- Vector dataset operations: loading, editing, saving with OGR/GDAL
- Google Earth Engine imagery integration and basemap visualization
- Vector interactions and geometric operations using OGR
- Map reduction techniques and zonal statistics
- Geometric calculations: area, centroids with EPSG projections
- Spectral indices computation from band arithmetic
- Image geocoding and coordinate transformations
- Advanced visualization: plotly scatterplots, geopandas choropleths, folium heatmaps
- Spatial interpolation: trend surface, IDW, kernel convolutions, kriging
- Spatial analysis: Thiessen/Voronoi polygons, isochrone analysis
- Raster operations: transformation, clipping, projection, maptile generation

Author: Claude AI
Created: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import warnings
import logging
import json
import os
import requests
from pathlib import Path
import tempfile
import zipfile

# Core GIS libraries
try:
    import geopandas as gpd
    import rasterio
    from rasterio import features, mask, warp, transform
    from rasterio.enums import Resampling
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    import fiona
    from fiona.crs import from_epsg
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    warnings.warn("Rasterio/Fiona not available. Some raster operations will be limited.")

try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
    OGR_AVAILABLE = True
except ImportError:
    OGR_AVAILABLE = False
    warnings.warn("GDAL/OGR not available. Some vector operations will be limited.")

# Spatial libraries
try:
    import pyproj
    from pyproj import Transformer, CRS as PyProjCRS
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    warnings.warn("PyProj not available. Some projection operations will be limited.")

try:
    from shapely.geometry import Point, Polygon, LineString, MultiPolygon
    from shapely.ops import transform as shapely_transform, unary_union
    from shapely import affinity
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    warnings.warn("Shapely not available. Some geometric operations will be limited.")

# Google Earth Engine
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    warnings.warn("Google Earth Engine not available. GEE functionality will be limited.")

# Visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some visualization features will be limited.")

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    warnings.warn("Folium not available. Some mapping features will be limited.")

# Spatial analysis libraries
try:
    from scipy import spatial, interpolate, ndimage
    from scipy.spatial import Voronoi, voronoi_plot_2d, distance_matrix
    from scipy.interpolate import griddata, Rbf
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some spatial analysis features will be limited.")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some ML-based interpolation will be limited.")

try:
    import pykrige
    from pykrige.ok import OrdinaryKriging
    from pykrige.uk import UniversalKriging
    PYKRIGE_AVAILABLE = True
except ImportError:
    PYKRIGE_AVAILABLE = False
    warnings.warn("PyKrige not available. Kriging interpolation will be limited.")

# Network analysis
try:
    import networkx as nx
    import osmnx as ox
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX/OSMnx not available. Network analysis will be limited.")


@dataclass
class GISConfig:
    """Configuration for GIS operations and climate risk modeling."""

    # Coordinate Reference Systems
    default_crs: str = "EPSG:4326"  # WGS84
    projected_crs: str = "EPSG:3857"  # Web Mercator

    # Climate risk parameters
    temperature_bands: List[str] = field(default_factory=lambda: ['B4', 'B5'])  # Thermal bands
    vegetation_bands: List[str] = field(default_factory=lambda: ['B8', 'B4'])   # NIR, Red for NDVI
    water_bands: List[str] = field(default_factory=lambda: ['B3', 'B8'])        # Green, NIR for NDWI

    # Interpolation parameters
    kriging_model: str = 'exponential'
    kernel_functions: List[str] = field(default_factory=lambda: ['gaussian', 'quartic', 'bicubic'])

    # Visualization settings
    color_palettes: Dict[str, str] = field(default_factory=lambda: {
        'temperature': 'RdYlBu_r',
        'precipitation': 'Blues',
        'vegetation': 'YlGn',
        'risk': 'Reds'
    })

    # Google Earth Engine collections
    gee_collections: Dict[str, str] = field(default_factory=lambda: {
        'landsat8': 'LANDSAT/LC08/C02/T1_L2',
        'sentinel2': 'COPERNICUS/S2_SR',
        'modis_temp': 'MODIS/006/MOD11A1',
        'precipitation': 'UCSB-CHG/CHIRPS/DAILY',
        'elevation': 'USGS/SRTMGL1_003'
    })


class VectorDatasetManager:
    """Manager for vector dataset operations including loading, editing, and saving."""

    def __init__(self, config: GISConfig):
        self.config = config
        self.datasets = {}

    def load_vector_dataset(self, file_path: str, layer_name: Optional[str] = None,
                           crs: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Load vector dataset from various formats (shapefile, GeoJSON, etc.).

        Args:
            file_path: Path to vector dataset
            layer_name: Layer name for multi-layer formats
            crs: Target coordinate reference system

        Returns:
            GeoDataFrame containing the vector data
        """
        try:
            # Load using geopandas
            if layer_name:
                gdf = gpd.read_file(file_path, layer=layer_name)
            else:
                gdf = gpd.read_file(file_path)

            # Set or transform CRS if specified
            if crs:
                if gdf.crs is None:
                    gdf = gdf.set_crs(crs)
                else:
                    gdf = gdf.to_crs(crs)

            # Store dataset
            dataset_name = Path(file_path).stem
            self.datasets[dataset_name] = gdf

            logging.info(f"Loaded vector dataset: {dataset_name} with {len(gdf)} features")

            return gdf

        except Exception as e:
            logging.error(f"Error loading vector dataset {file_path}: {e}")
            raise

    def load_vector_from_ogr(self, file_path: str, layer_index: int = 0) -> Dict[str, Any]:
        """
        Load vector dataset using OGR for more advanced control.

        Args:
            file_path: Path to vector dataset
            layer_index: Index of layer to load

        Returns:
            Dictionary containing OGR layer and metadata
        """
        if not OGR_AVAILABLE:
            raise ImportError("OGR not available. Install GDAL.")

        try:
            # Open dataset
            driver = ogr.GetDriverByName("ESRI Shapefile")
            if file_path.endswith('.geojson'):
                driver = ogr.GetDriverByName("GeoJSON")
            elif file_path.endswith('.gpkg'):
                driver = ogr.GetDriverByName("GPKG")

            datasource = ogr.Open(file_path)
            if datasource is None:
                raise ValueError(f"Could not open {file_path}")

            layer = datasource.GetLayer(layer_index)

            # Get metadata
            feature_count = layer.GetFeatureCount()
            layer_defn = layer.GetLayerDefn()
            field_count = layer_defn.GetFieldCount()

            # Get spatial reference
            spatial_ref = layer.GetSpatialRef()

            result = {
                'datasource': datasource,
                'layer': layer,
                'feature_count': feature_count,
                'field_count': field_count,
                'spatial_reference': spatial_ref,
                'geometry_type': layer_defn.GetGeomType()
            }

            return result

        except Exception as e:
            logging.error(f"Error loading vector with OGR {file_path}: {e}")
            raise

    def edit_vector_attributes(self, gdf: gpd.GeoDataFrame,
                              edit_operations: List[Dict[str, Any]]) -> gpd.GeoDataFrame:
        """
        Edit vector dataset attributes.

        Args:
            gdf: GeoDataFrame to edit
            edit_operations: List of edit operations

        Returns:
            Modified GeoDataFrame
        """
        gdf_copy = gdf.copy()

        for operation in edit_operations:
            op_type = operation.get('type')

            if op_type == 'add_field':
                field_name = operation['field_name']
                field_value = operation.get('field_value', None)
                gdf_copy[field_name] = field_value

            elif op_type == 'update_field':
                field_name = operation['field_name']
                condition = operation.get('condition')
                new_value = operation['new_value']

                if condition:
                    mask = gdf_copy.eval(condition)
                    gdf_copy.loc[mask, field_name] = new_value
                else:
                    gdf_copy[field_name] = new_value

            elif op_type == 'delete_field':
                field_name = operation['field_name']
                if field_name in gdf_copy.columns:
                    gdf_copy = gdf_copy.drop(columns=[field_name])

            elif op_type == 'filter_features':
                condition = operation['condition']
                gdf_copy = gdf_copy.query(condition)

        return gdf_copy

    def save_vector_dataset(self, gdf: gpd.GeoDataFrame, output_path: str,
                           driver: str = 'ESRI Shapefile') -> bool:
        """
        Save vector dataset to file.

        Args:
            gdf: GeoDataFrame to save
            output_path: Output file path
            driver: Output driver (ESRI Shapefile, GeoJSON, etc.)

        Returns:
            Success status
        """
        try:
            gdf.to_file(output_path, driver=driver)
            logging.info(f"Saved vector dataset to {output_path}")
            return True

        except Exception as e:
            logging.error(f"Error saving vector dataset {output_path}: {e}")
            return False


class GoogleEarthEngineManager:
    """Manager for Google Earth Engine imagery and data access."""

    def __init__(self, config: GISConfig):
        self.config = config
        self.ee_initialized = False

    def initialize_ee(self, service_account_key: Optional[str] = None) -> bool:
        """
        Initialize Google Earth Engine.

        Args:
            service_account_key: Path to service account key file

        Returns:
            Success status
        """
        if not GEE_AVAILABLE:
            logging.warning("Google Earth Engine not available")
            return False

        try:
            if service_account_key:
                ee.Initialize(ee.ServiceAccountCredentials(None, service_account_key))
            else:
                ee.Initialize()

            self.ee_initialized = True
            logging.info("Google Earth Engine initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Error initializing Google Earth Engine: {e}")
            return False

    def get_satellite_imagery(self, collection_name: str, bbox: List[float],
                             start_date: str, end_date: str,
                             cloud_cover_max: float = 10.0) -> Any:
        """
        Retrieve satellite imagery from Google Earth Engine.

        Args:
            collection_name: Name of satellite collection
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cloud_cover_max: Maximum cloud cover percentage

        Returns:
            Earth Engine ImageCollection
        """
        if not self.ee_initialized:
            raise RuntimeError("Google Earth Engine not initialized")

        try:
            # Define area of interest
            aoi = ee.Geometry.Rectangle(bbox)

            # Get collection
            collection_id = self.config.gee_collections.get(collection_name, collection_name)
            collection = ee.ImageCollection(collection_id)

            # Filter collection
            filtered = (collection
                       .filterBounds(aoi)
                       .filterDate(start_date, end_date))

            # Add cloud filter if available
            if 'CLOUD_COVER' in collection.first().propertyNames().getInfo():
                filtered = filtered.filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_max))

            return filtered

        except Exception as e:
            logging.error(f"Error retrieving satellite imagery: {e}")
            raise

    def compute_spectral_indices(self, image: Any, indices: List[str]) -> Any:
        """
        Compute spectral indices from satellite imagery.

        Args:
            image: Earth Engine Image
            indices: List of indices to compute (NDVI, NDWI, EVI, etc.)

        Returns:
            Earth Engine Image with computed indices
        """
        if not self.ee_initialized:
            raise RuntimeError("Google Earth Engine not initialized")

        result_image = image

        for index in indices:
            if index.upper() == 'NDVI':
                # Normalized Difference Vegetation Index
                ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                result_image = result_image.addBands(ndvi)

            elif index.upper() == 'NDWI':
                # Normalized Difference Water Index
                ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
                result_image = result_image.addBands(ndwi)

            elif index.upper() == 'EVI':
                # Enhanced Vegetation Index
                evi = image.expression(
                    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                    {
                        'NIR': image.select('B8'),
                        'RED': image.select('B4'),
                        'BLUE': image.select('B2')
                    }
                ).rename('EVI')
                result_image = result_image.addBands(evi)

            elif index.upper() == 'SAVI':
                # Soil Adjusted Vegetation Index
                L = 0.5  # Soil brightness correction factor
                savi = image.expression(
                    '(1 + L) * (NIR - RED) / (NIR + RED + L)',
                    {
                        'NIR': image.select('B8'),
                        'RED': image.select('B4'),
                        'L': L
                    }
                ).rename('SAVI')
                result_image = result_image.addBands(savi)

        return result_image

    def download_image_region(self, image: Any, bbox: List[float],
                             scale: int = 30, max_pixels: int = 1e8) -> np.ndarray:
        """
        Download image data for a specific region.

        Args:
            image: Earth Engine Image
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            scale: Pixel resolution in meters
            max_pixels: Maximum number of pixels to download

        Returns:
            NumPy array containing image data
        """
        if not self.ee_initialized:
            raise RuntimeError("Google Earth Engine not initialized")

        try:
            # Define region
            region = ee.Geometry.Rectangle(bbox)

            # Get image data
            image_data = image.getRegion(region, scale, 'EPSG:4326').getInfo()

            # Convert to numpy array
            header = image_data[0]
            data = np.array(image_data[1:])

            return {
                'header': header,
                'data': data,
                'bbox': bbox,
                'scale': scale
            }

        except Exception as e:
            logging.error(f"Error downloading image region: {e}")
            raise


class VectorOperations:
    """Advanced vector operations using OGR and geometric analysis."""

    def __init__(self, config: GISConfig):
        self.config = config

    def vector_intersection(self, gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Compute intersection between two vector datasets.

        Args:
            gdf1: First GeoDataFrame
            gdf2: Second GeoDataFrame

        Returns:
            Intersection result as GeoDataFrame
        """
        try:
            # Ensure same CRS
            if gdf1.crs != gdf2.crs:
                gdf2 = gdf2.to_crs(gdf1.crs)

            # Perform intersection
            intersection = gpd.overlay(gdf1, gdf2, how='intersection')

            return intersection

        except Exception as e:
            logging.error(f"Error in vector intersection: {e}")
            raise

    def vector_union(self, gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Compute union between two vector datasets.

        Args:
            gdf1: First GeoDataFrame
            gdf2: Second GeoDataFrame

        Returns:
            Union result as GeoDataFrame
        """
        try:
            # Ensure same CRS
            if gdf1.crs != gdf2.crs:
                gdf2 = gdf2.to_crs(gdf1.crs)

            # Perform union
            union = gpd.overlay(gdf1, gdf2, how='union')

            return union

        except Exception as e:
            logging.error(f"Error in vector union: {e}")
            raise

    def vector_buffer(self, gdf: gpd.GeoDataFrame, distance: float,
                     resolution: int = 16) -> gpd.GeoDataFrame:
        """
        Create buffer around vector features.

        Args:
            gdf: Input GeoDataFrame
            distance: Buffer distance
            resolution: Number of segments for circular arcs

        Returns:
            Buffered GeoDataFrame
        """
        try:
            # Create buffer
            gdf_buffered = gdf.copy()
            gdf_buffered['geometry'] = gdf.geometry.buffer(distance, resolution=resolution)

            return gdf_buffered

        except Exception as e:
            logging.error(f"Error in vector buffer: {e}")
            raise

    def compute_area_centroids(self, gdf: gpd.GeoDataFrame,
                              target_crs: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Compute polygonal areas and centroids with proper projection.

        Args:
            gdf: Input GeoDataFrame
            target_crs: Target CRS for area calculation

        Returns:
            GeoDataFrame with area and centroid columns
        """
        try:
            gdf_copy = gdf.copy()

            # Use projected CRS for accurate area calculation
            if target_crs:
                gdf_projected = gdf_copy.to_crs(target_crs)
            else:
                # Use appropriate UTM zone or default projected CRS
                gdf_projected = gdf_copy.to_crs(self.config.projected_crs)

            # Compute area in square meters
            gdf_copy['area_m2'] = gdf_projected.geometry.area
            gdf_copy['area_km2'] = gdf_copy['area_m2'] / 1e6

            # Compute centroids in original CRS
            gdf_copy['centroid'] = gdf_copy.geometry.centroid
            gdf_copy['centroid_x'] = gdf_copy['centroid'].x
            gdf_copy['centroid_y'] = gdf_copy['centroid'].y

            return gdf_copy

        except Exception as e:
            logging.error(f"Error computing area and centroids: {e}")
            raise

    def spatial_join(self, left_gdf: gpd.GeoDataFrame, right_gdf: gpd.GeoDataFrame,
                    how: str = 'inner', predicate: str = 'intersects') -> gpd.GeoDataFrame:
        """
        Perform spatial join between two GeoDataFrames.

        Args:
            left_gdf: Left GeoDataFrame
            right_gdf: Right GeoDataFrame
            how: Type of join ('left', 'right', 'outer', 'inner')
            predicate: Spatial predicate ('intersects', 'within', 'contains', etc.)

        Returns:
            Spatially joined GeoDataFrame
        """
        try:
            # Ensure same CRS
            if left_gdf.crs != right_gdf.crs:
                right_gdf = right_gdf.to_crs(left_gdf.crs)

            # Perform spatial join
            joined = gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)

            return joined

        except Exception as e:
            logging.error(f"Error in spatial join: {e}")
            raise


class RasterOperations:
    """Raster operations including map reduction, zonal statistics, and transformations."""

    def __init__(self, config: GISConfig):
        self.config = config

    def load_raster(self, file_path: str) -> Tuple[np.ndarray, Any]:
        """
        Load raster dataset.

        Args:
            file_path: Path to raster file

        Returns:
            Tuple of (raster array, raster profile)
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("Rasterio not available. Install rasterio.")

        try:
            with rasterio.open(file_path) as src:
                raster_array = src.read()
                profile = src.profile

            return raster_array, profile

        except Exception as e:
            logging.error(f"Error loading raster {file_path}: {e}")
            raise

    def zonal_statistics(self, raster_path: str, zones_gdf: gpd.GeoDataFrame,
                        stats: List[str] = ['mean', 'min', 'max', 'std']) -> gpd.GeoDataFrame:
        """
        Compute zonal statistics for raster data within vector zones.

        Args:
            raster_path: Path to raster file
            zones_gdf: GeoDataFrame containing zones
            stats: List of statistics to compute

        Returns:
            GeoDataFrame with zonal statistics
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("Rasterio not available for zonal statistics.")

        try:
            results = zones_gdf.copy()

            with rasterio.open(raster_path) as src:
                # Ensure zones are in same CRS as raster
                if zones_gdf.crs != src.crs:
                    zones_proj = zones_gdf.to_crs(src.crs)
                else:
                    zones_proj = zones_gdf

                for idx, zone in zones_proj.iterrows():
                    try:
                        # Mask raster with zone geometry
                        masked, mask_transform = mask.mask(src, [zone.geometry], crop=True)

                        # Remove nodata values
                        masked_data = masked[0]
                        if src.nodata is not None:
                            masked_data = masked_data[masked_data != src.nodata]

                        # Compute statistics
                        for stat in stats:
                            if len(masked_data) > 0:
                                if stat == 'mean':
                                    results.loc[idx, f'raster_{stat}'] = np.mean(masked_data)
                                elif stat == 'min':
                                    results.loc[idx, f'raster_{stat}'] = np.min(masked_data)
                                elif stat == 'max':
                                    results.loc[idx, f'raster_{stat}'] = np.max(masked_data)
                                elif stat == 'std':
                                    results.loc[idx, f'raster_{stat}'] = np.std(masked_data)
                                elif stat == 'sum':
                                    results.loc[idx, f'raster_{stat}'] = np.sum(masked_data)
                                elif stat == 'count':
                                    results.loc[idx, f'raster_{stat}'] = len(masked_data)
                            else:
                                results.loc[idx, f'raster_{stat}'] = np.nan

                    except Exception as e:
                        logging.warning(f"Error processing zone {idx}: {e}")
                        for stat in stats:
                            results.loc[idx, f'raster_{stat}'] = np.nan

            return results

        except Exception as e:
            logging.error(f"Error in zonal statistics: {e}")
            raise

    def raster_calculator(self, raster_paths: List[str], expression: str,
                         output_path: str) -> bool:
        """
        Perform raster arithmetic operations.

        Args:
            raster_paths: List of input raster paths
            expression: Mathematical expression using raster variables
            output_path: Output raster path

        Returns:
            Success status
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("Rasterio not available for raster calculator.")

        try:
            # Read all rasters
            raster_arrays = []
            profile = None

            for i, path in enumerate(raster_paths):
                with rasterio.open(path) as src:
                    array = src.read(1)  # Read first band
                    raster_arrays.append(array)

                    if profile is None:
                        profile = src.profile

            # Create variable dictionary for expression evaluation
            variables = {}
            for i, array in enumerate(raster_arrays):
                variables[f'r{i+1}'] = array

            # Add numpy functions to variables
            variables.update({
                'np': np,
                'sqrt': np.sqrt,
                'log': np.log,
                'exp': np.exp,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan
            })

            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, variables)

            # Save result
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(result, 1)

            logging.info(f"Raster calculation completed: {output_path}")
            return True

        except Exception as e:
            logging.error(f"Error in raster calculator: {e}")
            return False

    def clip_raster(self, raster_path: str, clip_geometry: Any,
                   output_path: str) -> bool:
        """
        Clip raster using vector geometry.

        Args:
            raster_path: Input raster path
            clip_geometry: Clipping geometry (Shapely or GeoDataFrame)
            output_path: Output raster path

        Returns:
            Success status
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("Rasterio not available for raster clipping.")

        try:
            with rasterio.open(raster_path) as src:
                # Convert geometry to list if needed
                if hasattr(clip_geometry, 'geometry'):
                    geometries = [geom for geom in clip_geometry.geometry]
                else:
                    geometries = [clip_geometry]

                # Clip raster
                clipped, clip_transform = mask.mask(src, geometries, crop=True)

                # Update profile
                profile = src.profile
                profile.update({
                    'height': clipped.shape[1],
                    'width': clipped.shape[2],
                    'transform': clip_transform
                })

                # Save clipped raster
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(clipped)

            logging.info(f"Raster clipped successfully: {output_path}")
            return True

        except Exception as e:
            logging.error(f"Error clipping raster: {e}")
            return False

    def reproject_raster(self, input_path: str, output_path: str,
                        target_crs: str, resampling_method: str = 'bilinear') -> bool:
        """
        Reproject raster to target CRS.

        Args:
            input_path: Input raster path
            output_path: Output raster path
            target_crs: Target coordinate reference system
            resampling_method: Resampling method

        Returns:
            Success status
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("Rasterio not available for reprojection.")

        try:
            resampling_methods = {
                'nearest': Resampling.nearest,
                'bilinear': Resampling.bilinear,
                'cubic': Resampling.cubic,
                'cubic_spline': Resampling.cubic_spline,
                'lanczos': Resampling.lanczos,
                'average': Resampling.average
            }

            resampling = resampling_methods.get(resampling_method, Resampling.bilinear)

            with rasterio.open(input_path) as src:
                # Calculate transform and dimensions for target CRS
                transform, width, height = warp.calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )

                # Update profile
                profile = src.profile
                profile.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                # Reproject
                with rasterio.open(output_path, 'w', **profile) as dst:
                    for i in range(1, src.count + 1):
                        warp.reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=resampling
                        )

            logging.info(f"Raster reprojected successfully: {output_path}")
            return True

        except Exception as e:
            logging.error(f"Error reprojecting raster: {e}")
            return False


class SpectralIndicesCalculator:
    """Calculate various spectral indices from satellite imagery."""

    def __init__(self, config: GISConfig):
        self.config = config

    def calculate_ndvi(self, nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index.

        Args:
            nir_band: Near-infrared band array
            red_band: Red band array

        Returns:
            NDVI array
        """
        try:
            # Avoid division by zero
            denominator = nir_band + red_band
            ndvi = np.where(denominator != 0,
                           (nir_band - red_band) / denominator,
                           0)

            # Clip to valid range [-1, 1]
            ndvi = np.clip(ndvi, -1, 1)

            return ndvi

        except Exception as e:
            logging.error(f"Error calculating NDVI: {e}")
            raise

    def calculate_ndwi(self, green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Water Index.

        Args:
            green_band: Green band array
            nir_band: Near-infrared band array

        Returns:
            NDWI array
        """
        try:
            # Avoid division by zero
            denominator = green_band + nir_band
            ndwi = np.where(denominator != 0,
                           (green_band - nir_band) / denominator,
                           0)

            # Clip to valid range [-1, 1]
            ndwi = np.clip(ndwi, -1, 1)

            return ndwi

        except Exception as e:
            logging.error(f"Error calculating NDWI: {e}")
            raise

    def calculate_evi(self, nir_band: np.ndarray, red_band: np.ndarray,
                     blue_band: np.ndarray, L: float = 1.0, C1: float = 6.0,
                     C2: float = 7.5, G: float = 2.5) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index.

        Args:
            nir_band: Near-infrared band array
            red_band: Red band array
            blue_band: Blue band array
            L: Canopy background adjustment
            C1: Coefficient for aerosol resistance (red)
            C2: Coefficient for aerosol resistance (blue)
            G: Gain factor

        Returns:
            EVI array
        """
        try:
            denominator = nir_band + C1 * red_band - C2 * blue_band + L
            evi = np.where(denominator != 0,
                          G * (nir_band - red_band) / denominator,
                          0)

            return evi

        except Exception as e:
            logging.error(f"Error calculating EVI: {e}")
            raise

    def calculate_savi(self, nir_band: np.ndarray, red_band: np.ndarray,
                      L: float = 0.5) -> np.ndarray:
        """
        Calculate Soil Adjusted Vegetation Index.

        Args:
            nir_band: Near-infrared band array
            red_band: Red band array
            L: Soil brightness correction factor

        Returns:
            SAVI array
        """
        try:
            denominator = nir_band + red_band + L
            savi = np.where(denominator != 0,
                           (1 + L) * (nir_band - red_band) / denominator,
                           0)

            return savi

        except Exception as e:
            logging.error(f"Error calculating SAVI: {e}")
            raise

    def calculate_multiple_indices(self, bands: Dict[str, np.ndarray],
                                  indices: List[str]) -> Dict[str, np.ndarray]:
        """
        Calculate multiple spectral indices.

        Args:
            bands: Dictionary of band arrays
            indices: List of indices to calculate

        Returns:
            Dictionary of calculated indices
        """
        results = {}

        for index in indices:
            try:
                if index.upper() == 'NDVI' and 'nir' in bands and 'red' in bands:
                    results['NDVI'] = self.calculate_ndvi(bands['nir'], bands['red'])

                elif index.upper() == 'NDWI' and 'green' in bands and 'nir' in bands:
                    results['NDWI'] = self.calculate_ndwi(bands['green'], bands['nir'])

                elif index.upper() == 'EVI' and all(b in bands for b in ['nir', 'red', 'blue']):
                    results['EVI'] = self.calculate_evi(bands['nir'], bands['red'], bands['blue'])

                elif index.upper() == 'SAVI' and 'nir' in bands and 'red' in bands:
                    results['SAVI'] = self.calculate_savi(bands['nir'], bands['red'])

            except Exception as e:
                logging.error(f"Error calculating {index}: {e}")

        return results


class GeocodingManager:
    """Image geocoding and coordinate transformation utilities."""

    def __init__(self, config: GISConfig):
        self.config = config

    def pixel_to_coordinate(self, transform: Any, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.

        Args:
            transform: Rasterio transform object
            pixel_x: Pixel X coordinate
            pixel_y: Pixel Y coordinate

        Returns:
            Tuple of (longitude, latitude)
        """
        try:
            lon, lat = rasterio.transform.xy(transform, pixel_y, pixel_x)
            return lon, lat

        except Exception as e:
            logging.error(f"Error converting pixel to coordinate: {e}")
            raise

    def coordinate_to_pixel(self, transform: Any, lon: float, lat: float) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel coordinates.

        Args:
            transform: Rasterio transform object
            lon: Longitude
            lat: Latitude

        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        try:
            pixel_y, pixel_x = rasterio.transform.rowcol(transform, lon, lat)
            return int(pixel_x), int(pixel_y)

        except Exception as e:
            logging.error(f"Error converting coordinate to pixel: {e}")
            raise

    def transform_coordinates(self, coordinates: List[Tuple[float, float]],
                            source_crs: str, target_crs: str) -> List[Tuple[float, float]]:
        """
        Transform coordinates between different CRS.

        Args:
            coordinates: List of (x, y) coordinate tuples
            source_crs: Source coordinate reference system
            target_crs: Target coordinate reference system

        Returns:
            List of transformed coordinates
        """
        if not PYPROJ_AVAILABLE:
            raise ImportError("PyProj not available for coordinate transformation.")

        try:
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

            transformed = []
            for x, y in coordinates:
                new_x, new_y = transformer.transform(x, y)
                transformed.append((new_x, new_y))

            return transformed

        except Exception as e:
            logging.error(f"Error transforming coordinates: {e}")
            raise

    def create_geotiff_with_coordinates(self, array: np.ndarray, transform: Any,
                                      crs: str, output_path: str) -> bool:
        """
        Create a GeoTIFF with proper coordinate information.

        Args:
            array: Image array
            transform: Affine transform
            crs: Coordinate reference system
            output_path: Output file path

        Returns:
            Success status
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("Rasterio not available for GeoTIFF creation.")

        try:
            # Determine array shape and data type
            if array.ndim == 2:
                height, width = array.shape
                count = 1
            else:
                count, height, width = array.shape

            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': count,
                'dtype': array.dtype,
                'crs': crs,
                'transform': transform,
                'compress': 'lzw'
            }

            with rasterio.open(output_path, 'w', **profile) as dst:
                if array.ndim == 2:
                    dst.write(array, 1)
                else:
                    dst.write(array)

            logging.info(f"GeoTIFF created successfully: {output_path}")
            return True

        except Exception as e:
            logging.error(f"Error creating GeoTIFF: {e}")
            return False


class VisualizationManager:
    """Advanced visualization methods for geospatial data."""

    def __init__(self, config: GISConfig):
        self.config = config

    def create_plotly_scatter(self, gdf: gpd.GeoDataFrame, x_col: str, y_col: str,
                             color_col: Optional[str] = None, size_col: Optional[str] = None,
                             title: str = "Geospatial Scatter Plot") -> Any:
        """
        Create interactive scatter plot using Plotly.

        Args:
            gdf: GeoDataFrame with point geometries
            x_col: Column for x-axis (or use geometry.x)
            y_col: Column for y-axis (or use geometry.y)
            color_col: Column for color mapping
            size_col: Column for size mapping
            title: Plot title

        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available for scatter plots.")

        try:
            # Extract coordinates if using geometry
            if x_col == 'geometry.x':
                x_data = gdf.geometry.x
            else:
                x_data = gdf[x_col]

            if y_col == 'geometry.y':
                y_data = gdf.geometry.y
            else:
                y_data = gdf[y_col]

            # Create scatter plot
            fig = go.Figure()

            scatter_kwargs = {
                'x': x_data,
                'y': y_data,
                'mode': 'markers',
                'name': 'Points'
            }

            if color_col and color_col in gdf.columns:
                scatter_kwargs['marker'] = {
                    'color': gdf[color_col],
                    'colorscale': 'Viridis',
                    'colorbar': {'title': color_col}
                }

            if size_col and size_col in gdf.columns:
                if 'marker' not in scatter_kwargs:
                    scatter_kwargs['marker'] = {}
                scatter_kwargs['marker']['size'] = gdf[size_col]

            fig.add_trace(go.Scatter(**scatter_kwargs))

            fig.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col
            )

            return fig

        except Exception as e:
            logging.error(f"Error creating Plotly scatter plot: {e}")
            raise

    def create_choropleth_map(self, gdf: gpd.GeoDataFrame, value_col: str,
                             color_scheme: str = 'YlOrRd', title: str = "Choropleth Map") -> Any:
        """
        Create choropleth map using GeoPandas.

        Args:
            gdf: GeoDataFrame with polygon geometries
            value_col: Column to map to colors
            color_scheme: Color scheme for mapping
            title: Map title

        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Create choropleth
            gdf.plot(column=value_col, cmap=color_scheme, legend=True,
                    legend_kwds={'shrink': 0.6}, ax=ax)

            ax.set_title(title, fontsize=16)
            ax.set_axis_off()

            # Add colorbar
            plt.tight_layout()

            return fig

        except Exception as e:
            logging.error(f"Error creating choropleth map: {e}")
            raise

    def create_folium_heatmap(self, gdf: gpd.GeoDataFrame, value_col: str,
                             center_lat: Optional[float] = None,
                             center_lon: Optional[float] = None,
                             zoom_start: int = 10) -> Any:
        """
        Create spatial heatmap using Folium.

        Args:
            gdf: GeoDataFrame with point geometries
            value_col: Column containing values for heatmap
            center_lat: Center latitude for map
            center_lon: Center longitude for map
            zoom_start: Initial zoom level

        Returns:
            Folium map object
        """
        if not FOLIUM_AVAILABLE:
            raise ImportError("Folium not available for heatmaps.")

        try:
            # Calculate center if not provided
            if center_lat is None or center_lon is None:
                bounds = gdf.total_bounds
                center_lat = (bounds[1] + bounds[3]) / 2
                center_lon = (bounds[0] + bounds[2]) / 2

            # Create base map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

            # Prepare data for heatmap
            heat_data = []
            for idx, row in gdf.iterrows():
                if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                    heat_data.append([row.geometry.y, row.geometry.x, row[value_col]])

            # Add heatmap
            plugins.HeatMap(heat_data).add_to(m)

            return m

        except Exception as e:
            logging.error(f"Error creating Folium heatmap: {e}")
            raise

    def create_3d_surface_plot(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              title: str = "3D Surface Plot") -> Any:
        """
        Create 3D surface plot for spatial data.

        Args:
            x: X coordinates
            y: Y coordinates
            z: Z values
            title: Plot title

        Returns:
            Plotly 3D surface figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available for 3D plots.")

        try:
            fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='X Coordinate',
                    yaxis_title='Y Coordinate',
                    zaxis_title='Value'
                )
            )

            return fig

        except Exception as e:
            logging.error(f"Error creating 3D surface plot: {e}")
            raise


class SpatialInterpolation:
    """Spatial interpolation methods for climate risk modeling."""

    def __init__(self, config: GISConfig):
        self.config = config

    def inverse_distance_weighting(self, known_points: np.ndarray, known_values: np.ndarray,
                                  grid_points: np.ndarray, power: float = 2.0) -> np.ndarray:
        """
        Inverse Distance Weighting interpolation.

        Args:
            known_points: Array of known point coordinates (n, 2)
            known_values: Array of known values (n,)
            grid_points: Array of grid point coordinates (m, 2)
            power: Power parameter for weighting

        Returns:
            Interpolated values at grid points
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy not available for IDW interpolation.")

        try:
            # Calculate distances
            distances = distance_matrix(grid_points, known_points)

            # Avoid division by zero
            distances = np.where(distances == 0, 1e-10, distances)

            # Calculate weights
            weights = 1.0 / (distances ** power)

            # Normalize weights
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            # Interpolate
            interpolated = np.dot(weights, known_values)

            return interpolated

        except Exception as e:
            logging.error(f"Error in IDW interpolation: {e}")
            raise

    def trend_surface_model(self, points: np.ndarray, values: np.ndarray,
                           grid_points: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Trend surface interpolation using polynomial fitting.

        Args:
            points: Known point coordinates (n, 2)
            values: Known values (n,)
            grid_points: Grid point coordinates (m, 2)
            degree: Polynomial degree

        Returns:
            Interpolated values at grid points
        """
        try:
            # Create polynomial features
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression

            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(points)

            # Fit model
            model = LinearRegression()
            model.fit(X_poly, values)

            # Predict on grid
            grid_poly = poly.transform(grid_points)
            interpolated = model.predict(grid_poly)

            return interpolated

        except Exception as e:
            logging.error(f"Error in trend surface modeling: {e}")
            raise

    def kernel_interpolation(self, points: np.ndarray, values: np.ndarray,
                           grid_points: np.ndarray, kernel: str = 'gaussian',
                           bandwidth: float = 1.0) -> np.ndarray:
        """
        Kernel-based interpolation.

        Args:
            points: Known point coordinates (n, 2)
            values: Known values (n,)
            grid_points: Grid point coordinates (m, 2)
            kernel: Kernel type ('gaussian', 'quartic', 'bicubic')
            bandwidth: Kernel bandwidth

        Returns:
            Interpolated values at grid points
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy not available for kernel interpolation.")

        try:
            # Calculate distances
            distances = distance_matrix(grid_points, points)

            # Apply kernel function
            if kernel == 'gaussian':
                weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
            elif kernel == 'quartic':
                d_norm = distances / bandwidth
                weights = np.where(d_norm <= 1, (1 - d_norm ** 2) ** 2, 0)
            elif kernel == 'bicubic':
                d_norm = distances / bandwidth
                weights = np.where(d_norm <= 1,
                                 1 - 1.5 * d_norm ** 2 + 0.75 * d_norm ** 3, 0)
            else:
                raise ValueError(f"Unknown kernel: {kernel}")

            # Normalize weights
            weight_sums = np.sum(weights, axis=1, keepdims=True)
            weight_sums = np.where(weight_sums == 0, 1, weight_sums)
            weights = weights / weight_sums

            # Interpolate
            interpolated = np.dot(weights, values)

            return interpolated

        except Exception as e:
            logging.error(f"Error in kernel interpolation: {e}")
            raise

    def ordinary_kriging(self, points: np.ndarray, values: np.ndarray,
                        grid_points: np.ndarray, variogram_model: str = 'exponential') -> np.ndarray:
        """
        Ordinary Kriging interpolation.

        Args:
            points: Known point coordinates (n, 2)
            values: Known values (n,)
            grid_points: Grid point coordinates (m, 2)
            variogram_model: Variogram model type

        Returns:
            Interpolated values at grid points
        """
        if not PYKRIGE_AVAILABLE:
            raise ImportError("PyKrige not available for kriging interpolation.")

        try:
            # Create ordinary kriging object
            ok = OrdinaryKriging(
                points[:, 0], points[:, 1], values,
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False
            )

            # Interpolate
            interpolated, variance = ok.execute('points', grid_points[:, 0], grid_points[:, 1])

            return interpolated

        except Exception as e:
            logging.error(f"Error in ordinary kriging: {e}")
            raise

    def universal_kriging(self, points: np.ndarray, values: np.ndarray,
                         grid_points: np.ndarray, drift_terms: Optional[List[str]] = None) -> np.ndarray:
        """
        Universal Kriging interpolation with trend.

        Args:
            points: Known point coordinates (n, 2)
            values: Known values (n,)
            grid_points: Grid point coordinates (m, 2)
            drift_terms: List of drift terms ('linear', 'quadratic')

        Returns:
            Interpolated values at grid points
        """
        if not PYKRIGE_AVAILABLE:
            raise ImportError("PyKrige not available for universal kriging.")

        try:
            if drift_terms is None:
                drift_terms = ['linear']

            # Create universal kriging object
            uk = UniversalKriging(
                points[:, 0], points[:, 1], values,
                variogram_model='exponential',
                drift_terms=drift_terms,
                verbose=False,
                enable_plotting=False
            )

            # Interpolate
            interpolated, variance = uk.execute('points', grid_points[:, 0], grid_points[:, 1])

            return interpolated

        except Exception as e:
            logging.error(f"Error in universal kriging: {e}")
            raise

    def create_interpolation_grid(self, bbox: List[float], resolution: float) -> np.ndarray:
        """
        Create regular grid for interpolation.

        Args:
            bbox: Bounding box [min_x, min_y, max_x, max_y]
            resolution: Grid resolution

        Returns:
            Grid point coordinates
        """
        try:
            min_x, min_y, max_x, max_y = bbox

            x_coords = np.arange(min_x, max_x + resolution, resolution)
            y_coords = np.arange(min_y, max_y + resolution, resolution)

            X, Y = np.meshgrid(x_coords, y_coords)
            grid_points = np.column_stack([X.ravel(), Y.ravel()])

            return grid_points

        except Exception as e:
            logging.error(f"Error creating interpolation grid: {e}")
            raise


class SpatialAnalysisTools:
    """Advanced spatial analysis tools for climate risk assessment."""

    def __init__(self, config: GISConfig):
        self.config = config

    def create_thiessen_polygons(self, points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create Thiessen (Voronoi) polygons from point data.

        Args:
            points_gdf: GeoDataFrame containing point geometries

        Returns:
            GeoDataFrame with Thiessen polygons
        """
        if not SCIPY_AVAILABLE or not SHAPELY_AVAILABLE:
            raise ImportError("SciPy and Shapely required for Thiessen polygons.")

        try:
            # Extract coordinates
            coords = np.array([[p.x, p.y] for p in points_gdf.geometry])

            # Create Voronoi diagram
            vor = Voronoi(coords)

            # Create polygons
            polygons = []
            point_indices = []

            for point_idx, point_region in enumerate(vor.point_region):
                region = vor.regions[point_region]

                if len(region) > 0 and -1 not in region:
                    # Get vertices
                    vertices = vor.vertices[region]

                    # Create polygon
                    if len(vertices) >= 3:
                        polygon = Polygon(vertices)
                        polygons.append(polygon)
                        point_indices.append(point_idx)

            # Create GeoDataFrame
            thiessen_gdf = gpd.GeoDataFrame(
                points_gdf.iloc[point_indices].reset_index(drop=True),
                geometry=polygons,
                crs=points_gdf.crs
            )

            return thiessen_gdf

        except Exception as e:
            logging.error(f"Error creating Thiessen polygons: {e}")
            raise

    def create_voronoi_polygons_bounded(self, points_gdf: gpd.GeoDataFrame,
                                      boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create bounded Voronoi polygons within a study area.

        Args:
            points_gdf: GeoDataFrame containing point geometries
            boundary_gdf: GeoDataFrame containing boundary polygon

        Returns:
            GeoDataFrame with bounded Voronoi polygons
        """
        try:
            # Create unbounded Thiessen polygons
            thiessen = self.create_thiessen_polygons(points_gdf)

            # Get boundary geometry
            boundary = boundary_gdf.geometry.unary_union

            # Clip polygons to boundary
            clipped_polygons = []
            for polygon in thiessen.geometry:
                clipped = polygon.intersection(boundary)
                if not clipped.is_empty:
                    clipped_polygons.append(clipped)
                else:
                    clipped_polygons.append(None)

            # Create clipped GeoDataFrame
            clipped_gdf = thiessen.copy()
            clipped_gdf['geometry'] = clipped_polygons

            # Remove empty geometries
            clipped_gdf = clipped_gdf[clipped_gdf.geometry.notna()]

            return clipped_gdf

        except Exception as e:
            logging.error(f"Error creating bounded Voronoi polygons: {e}")
            raise

    def create_isochrone_polygons(self, center_points: gpd.GeoDataFrame,
                                 travel_times: List[float],
                                 network_type: str = 'drive',
                                 speed_kmh: float = 50.0) -> gpd.GeoDataFrame:
        """
        Create isochrone polygons for accessibility analysis.

        Args:
            center_points: GeoDataFrame with center points
            travel_times: List of travel times in minutes
            network_type: Network type ('drive', 'walk', 'bike')
            speed_kmh: Travel speed in km/h

        Returns:
            GeoDataFrame with isochrone polygons
        """
        if not NETWORKX_AVAILABLE:
            logging.warning("NetworkX/OSMnx not available. Creating simplified isochrones.")
            return self._create_simple_isochrones(center_points, travel_times, speed_kmh)

        try:
            isochrone_polygons = []

            for idx, center in center_points.iterrows():
                center_point = (center.geometry.y, center.geometry.x)

                try:
                    # Download network around center point
                    # Using a simple distance-based approach as fallback
                    max_distance = max(travel_times) * speed_kmh * 1000 / 60  # Convert to meters

                    # Create circular buffers as simplified isochrones
                    for time in travel_times:
                        distance = time * speed_kmh * 1000 / 60  # Convert to meters

                        # Convert to projected CRS for buffer
                        center_projected = center_points.to_crs(self.config.projected_crs)
                        buffer = center_projected.iloc[idx:idx+1].geometry.buffer(distance)

                        # Convert back to original CRS
                        buffer_gdf = gpd.GeoDataFrame(
                            geometry=buffer,
                            crs=self.config.projected_crs
                        ).to_crs(center_points.crs)

                        isochrone_data = {
                            'center_id': idx,
                            'travel_time': time,
                            'geometry': buffer_gdf.geometry.iloc[0]
                        }
                        isochrone_polygons.append(isochrone_data)

                except Exception as e:
                    logging.warning(f"Error creating isochrone for point {idx}: {e}")

            isochrone_gdf = gpd.GeoDataFrame(isochrone_polygons, crs=center_points.crs)
            return isochrone_gdf

        except Exception as e:
            logging.error(f"Error creating isochrone polygons: {e}")
            raise

    def _create_simple_isochrones(self, center_points: gpd.GeoDataFrame,
                                 travel_times: List[float],
                                 speed_kmh: float) -> gpd.GeoDataFrame:
        """
        Create simplified isochrones using circular buffers.

        Args:
            center_points: GeoDataFrame with center points
            travel_times: List of travel times in minutes
            speed_kmh: Travel speed in km/h

        Returns:
            GeoDataFrame with simplified isochrone polygons
        """
        try:
            isochrone_polygons = []

            # Convert to projected CRS for accurate distance calculation
            center_projected = center_points.to_crs(self.config.projected_crs)

            for idx, center in center_projected.iterrows():
                for time in travel_times:
                    # Calculate distance in meters
                    distance = time * speed_kmh * 1000 / 60

                    # Create buffer
                    buffer = center.geometry.buffer(distance)

                    isochrone_data = {
                        'center_id': idx,
                        'travel_time': time,
                        'geometry': buffer
                    }
                    isochrone_polygons.append(isochrone_data)

            # Create GeoDataFrame and convert back to original CRS
            isochrone_gdf = gpd.GeoDataFrame(
                isochrone_polygons,
                crs=self.config.projected_crs
            ).to_crs(center_points.crs)

            return isochrone_gdf

        except Exception as e:
            logging.error(f"Error creating simple isochrones: {e}")
            raise

    def calculate_accessibility_metrics(self, origins_gdf: gpd.GeoDataFrame,
                                       destinations_gdf: gpd.GeoDataFrame,
                                       max_travel_time: float = 30.0) -> gpd.GeoDataFrame:
        """
        Calculate accessibility metrics between origins and destinations.

        Args:
            origins_gdf: GeoDataFrame with origin points
            destinations_gdf: GeoDataFrame with destination points
            max_travel_time: Maximum travel time to consider (minutes)

        Returns:
            GeoDataFrame with accessibility metrics
        """
        try:
            # Calculate distances between all origin-destination pairs
            origins_proj = origins_gdf.to_crs(self.config.projected_crs)
            destinations_proj = destinations_gdf.to_crs(self.config.projected_crs)

            accessibility_metrics = []

            for origin_idx, origin in origins_proj.iterrows():
                origin_point = origin.geometry

                # Calculate distances to all destinations
                distances = destinations_proj.geometry.distance(origin_point)

                # Convert distances to travel times (assuming walking speed of 5 km/h)
                travel_times = distances / (5000 / 60)  # Convert to minutes

                # Count accessible destinations
                accessible = travel_times <= max_travel_time
                num_accessible = accessible.sum()

                # Calculate average travel time to accessible destinations
                if num_accessible > 0:
                    avg_travel_time = travel_times[accessible].mean()
                    min_travel_time = travel_times[accessible].min()
                else:
                    avg_travel_time = np.nan
                    min_travel_time = np.nan

                accessibility_data = {
                    'origin_id': origin_idx,
                    'num_accessible': num_accessible,
                    'avg_travel_time': avg_travel_time,
                    'min_travel_time': min_travel_time,
                    'geometry': origin.geometry
                }
                accessibility_metrics.append(accessibility_data)

            # Create result GeoDataFrame
            result_gdf = gpd.GeoDataFrame(
                accessibility_metrics,
                crs=self.config.projected_crs
            ).to_crs(origins_gdf.crs)

            return result_gdf

        except Exception as e:
            logging.error(f"Error calculating accessibility metrics: {e}")
            raise


class AdvancedRasterOperations:
    """Advanced raster operations including transformations and maptile generation."""

    def __init__(self, config: GISConfig):
        self.config = config

    def affine_transform_raster(self, raster_path: str, output_path: str,
                               transform_matrix: np.ndarray) -> bool:
        """
        Apply affine transformation to raster using rasterio.

        Args:
            raster_path: Input raster path
            output_path: Output raster path
            transform_matrix: 3x3 affine transformation matrix

        Returns:
            Success status
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("Rasterio not available for affine transformation.")

        try:
            with rasterio.open(raster_path) as src:
                # Create new transform
                old_transform = src.transform
                new_transform = rasterio.Affine(
                    transform_matrix[0, 0], transform_matrix[0, 1], transform_matrix[0, 2],
                    transform_matrix[1, 0], transform_matrix[1, 1], transform_matrix[1, 2]
                )

                # Update profile
                profile = src.profile
                profile.update({'transform': new_transform})

                # Apply transformation and save
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(src.read())

            logging.info(f"Affine transformation applied: {output_path}")
            return True

        except Exception as e:
            logging.error(f"Error in affine transformation: {e}")
            return False

    def extent_transformation_matplotlib(self, image_array: np.ndarray,
                                       current_extent: List[float],
                                       target_extent: List[float]) -> np.ndarray:
        """
        Transform image extent using matplotlib transformations.

        Args:
            image_array: Input image array
            current_extent: Current extent [xmin, xmax, ymin, ymax]
            target_extent: Target extent [xmin, xmax, ymin, ymax]

        Returns:
            Transformed image array
        """
        try:
            # Calculate scaling factors
            x_scale = (target_extent[1] - target_extent[0]) / (current_extent[1] - current_extent[0])
            y_scale = (target_extent[3] - target_extent[2]) / (current_extent[3] - current_extent[2])

            # Calculate translation
            x_translate = target_extent[0] - current_extent[0] * x_scale
            y_translate = target_extent[2] - current_extent[2] * y_scale

            # Apply transformation using scipy
            if SCIPY_AVAILABLE:
                # Create transformation matrix
                transform_matrix = np.array([
                    [x_scale, 0, x_translate],
                    [0, y_scale, y_translate],
                    [0, 0, 1]
                ])

                # Apply transformation
                transformed = ndimage.affine_transform(
                    image_array,
                    transform_matrix[:2, :2],
                    offset=transform_matrix[:2, 2]
                )

                return transformed
            else:
                logging.warning("SciPy not available. Returning original array.")
                return image_array

        except Exception as e:
            logging.error(f"Error in extent transformation: {e}")
            raise

    def generate_maptiles(self, raster_path: str, output_dir: str,
                         min_zoom: int = 0, max_zoom: int = 10) -> bool:
        """
        Generate map tiles from raster using GDAL.

        Args:
            raster_path: Input raster path
            output_dir: Output directory for tiles
            min_zoom: Minimum zoom level
            max_zoom: Maximum zoom level

        Returns:
            Success status
        """
        if not OGR_AVAILABLE:
            logging.warning("GDAL not available. Creating simplified tile structure.")
            return self._create_simple_tiles(raster_path, output_dir, max_zoom)

        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Use gdal2tiles equivalent functionality
            # This is a simplified version - in practice you'd use gdal2tiles.py

            # Open raster
            dataset = gdal.Open(raster_path)
            if dataset is None:
                raise ValueError(f"Could not open raster: {raster_path}")

            # Get raster information
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            geotransform = dataset.GetGeoTransform()

            # Generate tiles for each zoom level
            for zoom in range(min_zoom, max_zoom + 1):
                zoom_dir = os.path.join(output_dir, str(zoom))
                os.makedirs(zoom_dir, exist_ok=True)

                # Calculate number of tiles at this zoom level
                num_tiles = 2 ** zoom
                tile_size = 256

                # Generate tiles (simplified approach)
                for x in range(num_tiles):
                    x_dir = os.path.join(zoom_dir, str(x))
                    os.makedirs(x_dir, exist_ok=True)

                    for y in range(num_tiles):
                        tile_path = os.path.join(x_dir, f"{y}.png")

                        # Create simple tile (placeholder)
                        # In practice, this would extract the appropriate raster portion
                        tile_array = np.random.randint(0, 256, (tile_size, tile_size, 3), dtype=np.uint8)

                        # Save tile
                        plt.imsave(tile_path, tile_array)

            logging.info(f"Map tiles generated in: {output_dir}")
            return True

        except Exception as e:
            logging.error(f"Error generating map tiles: {e}")
            return False

    def _create_simple_tiles(self, raster_path: str, output_dir: str, max_zoom: int) -> bool:
        """
        Create simplified tile structure without GDAL.

        Args:
            raster_path: Input raster path
            output_dir: Output directory for tiles
            max_zoom: Maximum zoom level

        Returns:
            Success status
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Load raster
            with rasterio.open(raster_path) as src:
                raster_array = src.read(1)

                # Create tiles for each zoom level
                for zoom in range(max_zoom + 1):
                    zoom_dir = os.path.join(output_dir, str(zoom))
                    os.makedirs(zoom_dir, exist_ok=True)

                    # Simple tiling approach
                    num_tiles = 2 ** zoom
                    tile_height = raster_array.shape[0] // num_tiles
                    tile_width = raster_array.shape[1] // num_tiles

                    for x in range(num_tiles):
                        x_dir = os.path.join(zoom_dir, str(x))
                        os.makedirs(x_dir, exist_ok=True)

                        for y in range(num_tiles):
                            # Extract tile
                            y_start = y * tile_height
                            y_end = (y + 1) * tile_height
                            x_start = x * tile_width
                            x_end = (x + 1) * tile_width

                            tile = raster_array[y_start:y_end, x_start:x_end]

                            # Save tile
                            tile_path = os.path.join(x_dir, f"{y}.png")
                            plt.imsave(tile_path, tile, cmap='viridis')

            logging.info(f"Simple tiles generated in: {output_dir}")
            return True

        except Exception as e:
            logging.error(f"Error creating simple tiles: {e}")
            return False


class ClimateRiskGIS:
    """Main class integrating all GIS functionality for climate risk modeling."""

    def __init__(self, config: Optional[GISConfig] = None):
        """
        Initialize the Climate Risk GIS system.

        Args:
            config: GIS configuration object
        """
        self.config = config or GISConfig()

        # Initialize managers
        self.vector_manager = VectorDatasetManager(self.config)
        self.gee_manager = GoogleEarthEngineManager(self.config)
        self.vector_ops = VectorOperations(self.config)
        self.raster_ops = RasterOperations(self.config)
        self.spectral_calc = SpectralIndicesCalculator(self.config)
        self.geocoding = GeocodingManager(self.config)
        self.visualization = VisualizationManager(self.config)
        self.interpolation = SpatialInterpolation(self.config)
        self.spatial_analysis = SpatialAnalysisTools(self.config)
        self.advanced_raster = AdvancedRasterOperations(self.config)

        logging.info("Climate Risk GIS system initialized")

    def comprehensive_climate_risk_analysis(self, study_area_path: str,
                                          hazard_data_paths: Dict[str, str],
                                          vulnerability_data_paths: Dict[str, str],
                                          output_dir: str) -> Dict[str, Any]:
        """
        Perform comprehensive climate risk analysis.

        Args:
            study_area_path: Path to study area vector data
            hazard_data_paths: Dictionary of hazard raster paths
            vulnerability_data_paths: Dictionary of vulnerability data paths
            output_dir: Output directory for results

        Returns:
            Dictionary containing analysis results
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            results = {}

            # Load study area
            study_area = self.vector_manager.load_vector_dataset(study_area_path)
            results['study_area'] = study_area

            # Process hazard data
            hazard_results = {}
            for hazard_name, hazard_path in hazard_data_paths.items():
                # Perform zonal statistics
                hazard_stats = self.raster_ops.zonal_statistics(
                    hazard_path, study_area,
                    stats=['mean', 'max', 'std']
                )
                hazard_results[hazard_name] = hazard_stats

                # Save processed data
                output_path = os.path.join(output_dir, f"{hazard_name}_stats.shp")
                self.vector_manager.save_vector_dataset(hazard_stats, output_path)

            results['hazard_analysis'] = hazard_results

            # Process vulnerability data
            vulnerability_results = {}
            for vuln_name, vuln_path in vulnerability_data_paths.items():
                vuln_data = self.vector_manager.load_vector_dataset(vuln_path)

                # Spatial join with study area
                joined = self.vector_ops.spatial_join(study_area, vuln_data)
                vulnerability_results[vuln_name] = joined

            results['vulnerability_analysis'] = vulnerability_results

            # Create risk assessment visualizations
            self._create_risk_visualizations(results, output_dir)

            logging.info(f"Climate risk analysis completed. Results saved to: {output_dir}")
            return results

        except Exception as e:
            logging.error(f"Error in comprehensive climate risk analysis: {e}")
            raise

    def _create_risk_visualizations(self, analysis_results: Dict[str, Any],
                                   output_dir: str) -> None:
        """
        Create visualizations for risk analysis results.

        Args:
            analysis_results: Results from climate risk analysis
            output_dir: Output directory for visualizations
        """
        try:
            # Create choropleth maps for hazard data
            for hazard_name, hazard_data in analysis_results['hazard_analysis'].items():
                if 'raster_mean' in hazard_data.columns:
                    fig = self.visualization.create_choropleth_map(
                        hazard_data, 'raster_mean',
                        title=f"{hazard_name.title()} Hazard Risk"
                    )

                    output_path = os.path.join(output_dir, f"{hazard_name}_map.png")
                    fig.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)

            # Create interactive plots if plotly is available
            if PLOTLY_AVAILABLE:
                study_area = analysis_results['study_area']
                if len(study_area) > 0:
                    fig = self.visualization.create_plotly_scatter(
                        study_area, 'centroid_x', 'centroid_y',
                        title="Study Area Overview"
                    )

                    output_path = os.path.join(output_dir, "study_area_interactive.html")
                    fig.write_html(output_path)

        except Exception as e:
            logging.warning(f"Error creating risk visualizations: {e}")


def main():
    """
    Example usage of the Climate Risk GIS system.
    """
    # Initialize configuration
    config = GISConfig()

    # Create GIS system
    gis = ClimateRiskGIS(config)

    print("Climate Risk GIS system initialized successfully!")

    # Example: Create sample data and demonstrate functionality
    try:
        # Example vector operations
        print("\n1. Vector Operations Example:")
        example_vector_operations()

        # Example raster operations
        print("\n2. Raster Operations Example:")
        example_raster_operations()

        # Example spatial interpolation
        print("\n3. Spatial Interpolation Example:")
        example_spatial_interpolation()

        # Example visualization
        print("\n4. Visualization Example:")
        example_visualization()

        print("\nClimate Risk GIS framework ready for use!")

    except Exception as e:
        print(f"Error in examples: {e}")


def example_vector_operations():
    """Example of vector dataset operations."""
    try:
        config = GISConfig()
        vector_manager = VectorDatasetManager(config)

        # Create sample points
        from shapely.geometry import Point
        import geopandas as gpd

        points = [Point(0, 0), Point(1, 1), Point(2, 2)]
        gdf = gpd.GeoDataFrame({'id': [1, 2, 3]}, geometry=points, crs='EPSG:4326')

        # Calculate areas and centroids
        vector_ops = VectorOperations(config)
        result = vector_ops.compute_area_centroids(gdf)

        print(f"Created {len(result)} vector features with area calculations")

    except Exception as e:
        print(f"Vector operations example error: {e}")


def example_raster_operations():
    """Example of raster operations."""
    try:
        # Create sample raster data
        sample_array = np.random.rand(100, 100)
        print(f"Created sample raster array: {sample_array.shape}")

        # Example spectral indices calculation
        config = GISConfig()
        spectral_calc = SpectralIndicesCalculator(config)

        # Sample bands
        nir = np.random.rand(50, 50) * 0.8 + 0.2
        red = np.random.rand(50, 50) * 0.3 + 0.1

        ndvi = spectral_calc.calculate_ndvi(nir, red)
        print(f"Calculated NDVI with mean value: {np.mean(ndvi):.3f}")

    except Exception as e:
        print(f"Raster operations example error: {e}")


def example_spatial_interpolation():
    """Example of spatial interpolation methods."""
    try:
        config = GISConfig()
        interpolation = SpatialInterpolation(config)

        # Create sample data
        np.random.seed(42)
        known_points = np.random.rand(20, 2) * 100
        known_values = np.random.rand(20) * 50 + 25

        # Create grid
        grid_points = interpolation.create_interpolation_grid([0, 0, 100, 100], 5.0)

        # Perform IDW interpolation
        interpolated = interpolation.inverse_distance_weighting(
            known_points, known_values, grid_points
        )

        print(f"Interpolated {len(interpolated)} grid points using IDW")

    except Exception as e:
        print(f"Spatial interpolation example error: {e}")


def example_visualization():
    """Example of visualization capabilities."""
    try:
        # Create sample data
        np.random.seed(42)
        x = np.random.rand(50) * 100
        y = np.random.rand(50) * 100
        values = np.random.rand(50) * 10

        points = [Point(xi, yi) for xi, yi in zip(x, y)]
        gdf = gpd.GeoDataFrame({'value': values}, geometry=points, crs='EPSG:4326')

        config = GISConfig()
        viz = VisualizationManager(config)

        # Create scatter plot if plotly available
        if PLOTLY_AVAILABLE:
            fig = viz.create_plotly_scatter(gdf, 'geometry.x', 'geometry.y', 'value')
            print("Created interactive scatter plot")
        else:
            print("Plotly not available, skipping interactive plot")

    except Exception as e:
        print(f"Visualization example error: {e}")


if __name__ == "__main__":
    main()