import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from parameters import Parameters


class Draw_netcdf:
    def __init__(self):
        parameters = Parameters()
        self.RESULT_PATH = parameters.RESULT_PATH
        self.SPATIAL_MAP_YEAR = parameters.SPATIAL_MAP_YEAR
        self.TREND_START_YEAR = parameters.TREND_START_YEAR
        self.TREND_END_YEAR = parameters.TREND_END_YEAR
        self.MIN_TREND_YEARS = 20

        if not os.path.exists(self.RESULT_PATH):
            os.makedirs(self.RESULT_PATH)

    def draw_spatial_anomaly_map(self, years, lat, lon, annual_maps):
        target_year = self._choose_available_year(years, self.SPATIAL_MAP_YEAR)
        year_index = np.where(years == target_year)[0][0]
        data = annual_maps[year_index]

        output_path = os.path.join(
            self.RESULT_PATH,
            f'Figure 2 - Global Temperature Anomaly Map {target_year}.png'
        )
        self._draw_map(
            data,
            lat,
            lon,
            title=f'Global Annual Mean Temperature Anomaly in {target_year}',
            colorbar_label='Temperature anomaly (deg C)',
            output_path=output_path,
            min_abs_limit=1.5,
        )

        print('Plot saved at ' + output_path)
        return output_path

    def draw_trend_map(self, years, lat, lon, annual_maps):
        start_year = max(self.TREND_START_YEAR, int(years.min()))
        end_year = min(self.TREND_END_YEAR, int(years.max()))
        period_mask = (years >= start_year) & (years <= end_year)
        period_years = years[period_mask]
        period_maps = annual_maps[period_mask]

        if len(period_years) < 2:
            raise ValueError('Not enough annual maps to calculate a trend.')

        trend = self._linear_trend_per_decade(period_years, period_maps)
        output_path = os.path.join(
            self.RESULT_PATH,
            f'Figure 3 - Temperature Trend Map {start_year}-{end_year}.png'
        )
        self._draw_map(
            trend,
            lat,
            lon,
            title=f'Linear Temperature Anomaly Trend, {start_year}-{end_year}',
            colorbar_label='Trend (deg C per decade)',
            output_path=output_path,
            min_abs_limit=0.2,
        )

        print('Plot saved at ' + output_path)
        return output_path

    def _choose_available_year(self, years, preferred_year):
        if preferred_year in years:
            return preferred_year
        latest_year = int(years.max())
        print(
            f'Year {preferred_year} is not complete in the grid data; '
            f'using {latest_year} instead.'
        )
        return latest_year

    def _linear_trend_per_decade(self, years, maps):
        x = years.astype(float)[:, None, None]
        y = maps.astype(float)
        valid = np.isfinite(y)
        count = valid.sum(axis=0)

        with np.errstate(invalid='ignore', divide='ignore'):
            x_sum = np.where(valid, x, 0.0).sum(axis=0)
            y_sum = np.where(valid, y, 0.0).sum(axis=0)
            x_mean = x_sum / count
            y_mean = y_sum / count

            dx = np.where(valid, x - x_mean, 0.0)
            dy = np.where(valid, y - y_mean, 0.0)
            denominator = (dx * dx).sum(axis=0)
            slope = (dx * dy).sum(axis=0) / denominator

        slope[count < self.MIN_TREND_YEARS] = np.nan
        return slope * 10.0

    def _draw_map(self, data, lat, lon, title, colorbar_label, output_path, min_abs_limit):
        plot_data, plot_lat, plot_lon = self._prepare_for_plot(data, lat, lon)
        vmin, vmax = self._symmetric_limits(plot_data, min_abs_limit)

        fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        image = ax.imshow(
            plot_data,
            extent=[plot_lon.min(), plot_lon.max(), plot_lat.min(), plot_lat.max()],
            origin='lower',
            cmap='RdBu_r',
            norm=norm,
            aspect='auto',
        )

        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.grid(color='black', alpha=0.15, linewidth=0.5)

        colorbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02)
        colorbar.set_label(colorbar_label)

        fig.tight_layout()
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    def _prepare_for_plot(self, data, lat, lon):
        plot_data = np.asarray(data)
        plot_lat = np.asarray(lat)
        plot_lon = np.asarray(lon)

        if plot_lat[0] > plot_lat[-1]:
            plot_lat = plot_lat[::-1]
            plot_data = plot_data[::-1, :]

        if plot_lon.min() >= 0:
            plot_lon = ((plot_lon + 180) % 360) - 180

        lon_order = np.argsort(plot_lon)
        plot_lon = plot_lon[lon_order]
        plot_data = plot_data[:, lon_order]
        return plot_data, plot_lat, plot_lon

    def _symmetric_limits(self, data, min_abs_limit):
        finite_values = data[np.isfinite(data)]
        if len(finite_values) == 0:
            return -min_abs_limit, min_abs_limit

        limit = np.nanpercentile(np.abs(finite_values), 98)
        limit = max(float(limit), min_abs_limit)
        return -limit, limit
