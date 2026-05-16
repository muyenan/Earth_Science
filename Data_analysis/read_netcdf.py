import gzip
import os
import struct
import warnings

import numpy as np

from parameters import Parameters


class NetCDFClassicReader:
    NC_DIMENSION = 10
    NC_VARIABLE = 11
    NC_ATTRIBUTE = 12

    TYPE_INFO = {
        1: ('>i1', 1),
        2: ('S1', 1),
        3: ('>i2', 2),
        4: ('>i4', 4),
        5: ('>f4', 4),
        6: ('>f8', 8),
    }

    def __init__(self, gz_path):
        self.gz_path = gz_path
        self.offset = 0
        self.dimensions = []
        self.global_attributes = {}
        self.variables = {}

        with gzip.open(gz_path, 'rb') as f:
            self.data = f.read()

        self._parse_header()

    def _read(self, size):
        value = self.data[self.offset:self.offset + size]
        self.offset += size
        return value

    def _read_int(self):
        return struct.unpack('>i', self._read(4))[0]

    def _read_uint(self):
        return struct.unpack('>I', self._read(4))[0]

    def _skip_padding(self, raw_size):
        padding = (4 - raw_size % 4) % 4
        if padding:
            self._read(padding)

    def _read_name(self):
        size = self._read_int()
        raw = self._read(size)
        self._skip_padding(size)
        return raw.decode('ascii')

    def _read_values(self, nc_type, count):
        dtype, item_size = self.TYPE_INFO[nc_type]
        raw_size = item_size * count
        raw = self._read(raw_size)
        self._skip_padding(raw_size)

        if nc_type == 2:
            return raw.decode('ascii').rstrip('\x00')

        values = np.frombuffer(raw, dtype=np.dtype(dtype)).copy()
        if count == 1:
            return values[0].item()
        return values

    def _read_attribute_list(self):
        tag = self._read_int()
        count = self._read_int()
        if tag == 0 and count == 0:
            return {}
        if tag != self.NC_ATTRIBUTE:
            raise ValueError('Invalid NetCDF attribute list.')

        attributes = {}
        for _ in range(count):
            name = self._read_name()
            nc_type = self._read_int()
            value_count = self._read_int()
            attributes[name] = self._read_values(nc_type, value_count)
        return attributes

    def _parse_header(self):
        magic = self._read(4)
        if magic != b'CDF\x01':
            raise ValueError('Only classic NetCDF CDF-1 files are supported.')

        self.num_records = self._read_int()
        self.dimensions = self._read_dimensions()
        self.global_attributes = self._read_attribute_list()
        self.variables = self._read_variables()

    def _read_dimensions(self):
        tag = self._read_int()
        count = self._read_int()
        if tag == 0 and count == 0:
            return []
        if tag != self.NC_DIMENSION:
            raise ValueError('Invalid NetCDF dimension list.')

        dimensions = []
        for _ in range(count):
            dimensions.append({
                'name': self._read_name(),
                'size': self._read_int(),
            })
        return dimensions

    def _read_variables(self):
        tag = self._read_int()
        count = self._read_int()
        if tag == 0 and count == 0:
            return {}
        if tag != self.NC_VARIABLE:
            raise ValueError('Invalid NetCDF variable list.')

        variables = {}
        for _ in range(count):
            name = self._read_name()
            dim_count = self._read_int()
            dim_ids = [self._read_int() for _ in range(dim_count)]
            attributes = self._read_attribute_list()
            nc_type = self._read_int()
            vsize = self._read_uint()
            begin = self._read_uint()

            shape = []
            is_record = False
            for index, dim_id in enumerate(dim_ids):
                size = self.dimensions[dim_id]['size']
                if index == 0 and size == 0:
                    size = self.num_records
                    is_record = True
                shape.append(size)

            variables[name] = {
                'name': name,
                'dim_ids': dim_ids,
                'shape': tuple(shape),
                'attributes': attributes,
                'nc_type': nc_type,
                'vsize': vsize,
                'begin': begin,
                'is_record': is_record,
            }
        return variables

    def read_variable(self, name):
        variable = self.variables[name]
        dtype, item_size = self.TYPE_INFO[variable['nc_type']]
        shape = variable['shape']

        if variable['is_record']:
            record_variables = [
                item for item in self.variables.values()
                if item['is_record']
            ]
            record_size = sum(item['vsize'] for item in record_variables)
            per_record_count = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            raw_size = per_record_count * item_size
            records = []

            for record_index in range(shape[0]):
                begin = variable['begin'] + record_index * record_size
                raw = self.data[begin:begin + raw_size]
                records.append(np.frombuffer(raw, dtype=np.dtype(dtype)).copy())

            values = np.stack(records, axis=0)
        else:
            count = int(np.prod(shape))
            raw_size = count * item_size
            begin = variable['begin']
            raw = self.data[begin:begin + raw_size]
            values = np.frombuffer(raw, dtype=np.dtype(dtype)).copy()

        values = values.reshape(shape)
        if variable['nc_type'] == 2:
            return values
        return self._apply_numeric_attributes(values, variable['attributes'])

    def _apply_numeric_attributes(self, values, attributes):
        raw_values = values
        values = values.astype(float)

        for attr_name in ('_FillValue', 'missing_value'):
            if attr_name in attributes:
                missing_values = np.atleast_1d(attributes[attr_name])
                for missing_value in missing_values:
                    values[raw_values == missing_value] = np.nan

        scale = float(attributes.get('scale_factor', 1.0))
        offset = float(attributes.get('add_offset', 0.0))
        return values * scale + offset


class Read_netcdf:
    def __init__(self):
        parameters = Parameters()
        self.NETCDF_GZ_PATH = parameters.NETCDF_GZ_PATH
        self.TEMPERATURE_VARIABLE = parameters.NETCDF_TEMPERATURE_VARIABLE
        self.LAT_VARIABLE = parameters.NETCDF_LAT_VARIABLE
        self.LON_VARIABLE = parameters.NETCDF_LON_VARIABLE
        self.FIRST_YEAR = parameters.NETCDF_FIRST_YEAR

    def read_annual_temperature_anomaly(self):
        if not os.path.exists(self.NETCDF_GZ_PATH):
            raise FileNotFoundError('File not found: ' + self.NETCDF_GZ_PATH)

        reader = NetCDFClassicReader(self.NETCDF_GZ_PATH)
        monthly_anomaly = reader.read_variable(self.TEMPERATURE_VARIABLE)
        lat = reader.read_variable(self.LAT_VARIABLE)
        lon = reader.read_variable(self.LON_VARIABLE)

        years, annual_maps = self._monthly_to_annual(monthly_anomaly)
        return years, lat, lon, annual_maps

    def _monthly_to_annual(self, monthly_anomaly):
        month_count = monthly_anomaly.shape[0]
        month_years = self.FIRST_YEAR + np.arange(month_count) // 12

        years = []
        annual_maps = []
        for year in np.unique(month_years):
            indexes = np.where(month_years == year)[0]
            if len(indexes) != 12:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                annual_map = np.nanmean(monthly_anomaly[indexes], axis=0)

            years.append(int(year))
            annual_maps.append(annual_map)

        return np.array(years), np.stack(annual_maps, axis=0)
