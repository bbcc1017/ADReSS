"""Offline source-data and hospital-identity regression check.

Run from the repository root: python tests/check_scenario_data.py
Routing responses are deterministic test values, not measured road travel times.
"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'sce_src'))
import make_csv_yaml_dynamic as generator_module


def main():
    master = pd.read_excel(ROOT / 'scenarios/hospital_master_data.xlsx')
    raw = pd.read_excel(ROOT / 'scenarios/DISTANCE_MATRIX_FINAL.xlsx',
                        sheet_name='Distance_Matrix', header=None)
    info = pd.read_excel(ROOT / 'scenarios/DISTANCE_MATRIX_FINAL.xlsx',
                         sheet_name='Hospital_Info')
    values = raw.iloc[1:, 1:].to_numpy(dtype=float)
    names = raw.iloc[0, 1:].tolist()
    assert values.shape == (len(master), len(master))
    assert names == raw.iloc[1:, 0].tolist() == info.institution_name.tolist()
    pd.testing.assert_frame_equal(info.drop(columns='Index'), master)
    assert np.isfinite(values).all() and (values >= 0).all()
    assert np.count_nonzero(np.diag(values)) == 0
    assert (values[~np.eye(len(values), dtype=bool)] > 0).all()
    assert not info.duplicated(['institution_name', 'x_coord', 'y_coord']).any()
    assert master.institution_name.duplicated().any(), 'Duplicate-name coverage required'

    key = lambda row: (row.institution_name, round(row.x_coord, 6), round(row.y_coord, 6))
    lookup = {key(row): i for i, row in info.iterrows()}
    generator = generator_module.ScenarioGenerator.__new__(generator_module.ScenarioGenerator)
    generator.base_path = str(ROOT)
    generator.hospital_data_path = str(ROOT / 'scenarios/hospital_master_data.xlsx')
    generator.patient_config = {'ratio': {'Red': .1, 'Yellow': .3, 'Green': .5, 'Black': .1}}
    generator.get_road_distance = lambda start, end, **kwargs: (
        generator_module.haversine(start, end), 40 - end[0])

    with tempfile.TemporaryDirectory(prefix='adress-data-check-') as folder:
        # A deliberately large target includes every hospital, including all repeated names.
        with redirect_stdout(StringIO()), patch.object(generator_module.time, 'sleep'):
            generator.make_hospital_info(37.45, 127.14, 1_000_000, folder, uav_count=3)
            generator.make_distance_Hos2Hos(folder)
            generator.make_uav_info(37.45, 127.14, 1_000_000, 3, folder)
        selected = pd.read_csv(Path(folder) / 'hospital_info_road.csv')
        assert len(selected) == len(master)
        indices = [lookup[key(row)] for _, row in selected.iterrows()]
        actual = pd.read_csv(Path(folder) / 'distance_Hos2Hos_road.csv', index_col=0).to_numpy()
        np.testing.assert_array_equal(actual, values[np.ix_(indices, indices)])
        euclidean = pd.read_csv(Path(folder) / 'distance_Hos2Hos_euc.csv', index_col=0).to_numpy()
        duplicates = selected.index[selected.institution_name.duplicated(False)].tolist()
        unique = selected.index[~selected.institution_name.duplicated(False)][0]
        for i in duplicates:
            for j in [unique, *duplicates]:
                a, b = selected.iloc[i], selected.iloc[j]
                expected = generator_module.haversine((a.y_coord, a.x_coord), (b.y_coord, b.x_coord))
                np.testing.assert_allclose(euclidean[i, j], expected)
        uavs = pd.read_csv(Path(folder) / 'uav_info.csv')
        assert len(uavs) == 3 and uavs.hospital_idx.is_unique
        assert (selected.loc[uavs.hospital_idx, 'helipad'] == 1).all()

    legacy = master.loc[~master.institution_name.duplicated(False), ['institution_name']].head(3)
    resolved = generator._with_hospital_coordinates(legacy)
    pd.testing.assert_frame_equal(
        resolved[['x_coord', 'y_coord']],
        master.loc[legacy.index, ['x_coord', 'y_coord']].reset_index(drop=True))
    ambiguous = master.loc[master.institution_name.duplicated(False), ['institution_name']].head(1)
    try:
        generator._with_hospital_coordinates(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError('Ambiguous legacy hospital name must not resolve silently')
    partial = master[['institution_name', 'x_coord']].head(1)
    try:
        generator._with_hospital_coordinates(partial)
    except ValueError:
        pass
    else:
        raise AssertionError('Partial coordinates must not be merged ambiguously')
    print(f'Checked {len(master)} hospitals, duplicate-name routing, matrix values, and UAV indices offline.')


if __name__ == '__main__':
    main()
