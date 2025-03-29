import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from preprocessing import DataPreprocessor  # Import your original DataPreprocessor

class TestDataPreprocessor:
    """Simple tests for DataPreprocessor focusing on core operations"""
    
    @pytest.fixture
    def sample_data(self):
        """Create a simple test dataset with typical features"""
        np.random.seed(42)
        n_samples = 500
        
        # Create DataFrame with key features
        df = pd.DataFrame({
            # Features for GeV scaling (in MeV)
            'min_PT_final_state_tracks': np.random.exponential(500, size=n_samples) + 250,
            'TwoBody_PT': np.random.exponential(1000, size=n_samples) + 1000,
            
            # Features for log transform
            'min_FS_OWNPVIPCHI2': np.random.exponential(100, size=n_samples) + 5,
            'TwoBody_OWNPVFDCHI2': np.random.exponential(1000, size=n_samples) + 20,
            
            # Feature without transformation
            'TwoBody_MAXDOCA': np.random.exponential(0.02, size=n_samples),
            
            # Required columns for processing
            'class_label': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
            'channel': np.random.choice(['minbias', 'signal1', 'signal2'], size=n_samples, p=[0.7, 0.15, 0.15])
        })
        
        return df
    
    @pytest.fixture
    def preprocessor(self):
        """Create a basic preprocessor"""
        gev_vars = ('min_PT_final_state_tracks', 'TwoBody_PT')
        log_vars = ('min_FS_OWNPVIPCHI2', 'TwoBody_OWNPVFDCHI2')
        unchanged_vars = ('TwoBody_MAXDOCA',)
        
        return DataPreprocessor(
            gev_vars=gev_vars, 
            log_vars=log_vars,
            unchanged_vars=unchanged_vars,
            clip_quantiles=(0.001, 0.999),
            normalize=True
        )
    
    def test_init(self, preprocessor):
        """Test that initialization sets up transformations correctly"""
        # Check transformation mapping
        assert preprocessor.transformations['min_PT_final_state_tracks'] == 'gev'
        assert preprocessor.transformations['TwoBody_PT'] == 'gev'
        assert preprocessor.transformations['min_FS_OWNPVIPCHI2'] == 'log'
        assert preprocessor.transformations['TwoBody_OWNPVFDCHI2'] == 'log'
        assert preprocessor.transformations['TwoBody_MAXDOCA'] == 'none'
    
    def test_balance_signal(self, preprocessor, sample_data):
        """Test signal balancing functionality"""
        balanced_df = preprocessor.balance_signal(sample_data)
        
        # Check that balanced dataframe has all channels
        assert set(balanced_df['channel'].unique()) == set(sample_data['channel'].unique())
        
        # Check that signal channels have equal counts
        signal_channels = [ch for ch in balanced_df['channel'].unique() if ch != 'minbias']
        signal_counts = balanced_df['channel'].value_counts()[signal_channels]
        assert len(signal_counts.unique()) == 1, "Signal channels should have equal representation"
        
        # Background channel should be unchanged
        assert balanced_df['channel'].value_counts()['minbias'] == sample_data['channel'].value_counts()['minbias']
    
    def test_fit_transform(self, preprocessor, sample_data):
        """Test the complete fit_transform pipeline"""
        # Apply fit_transform
        processed_df = preprocessor.fit_transform(sample_data)
        
        # 1. Check that GeV scaling works
        # Get the original min_PT and the transformed value for a sample
        sample_idx = 0
        original_pt = sample_data.loc[sample_idx, 'min_PT_final_state_tracks']
        transformed_pt = processed_df.loc[sample_idx, 'min_PT_final_state_tracks']
        
        # The transformed value should be in [0,1]
        assert 0 <= transformed_pt <= 1
        
        # 2. Check that log transformation works
        original_ipchi2 = sample_data.loc[sample_idx, 'min_FS_OWNPVIPCHI2']
        transformed_ipchi2 = processed_df.loc[sample_idx, 'min_FS_OWNPVIPCHI2']
        
        # The transformed value should be in [0,1]
        assert 0 <= transformed_ipchi2 <= 1
        
        # 3. Check clipping
        # Make a DataFrame with extreme values
        extreme_df = sample_data.copy()
        extreme_df.loc[0, 'min_PT_final_state_tracks'] = 1000000  # Very high PT
        extreme_df.loc[1, 'min_FS_OWNPVIPCHI2'] = 1000000  # Very high IPCHI2
        
        # Process with the already fitted preprocessor
        extreme_processed = preprocessor.transform(extreme_df)
        
        # Extreme values should be clipped to [0,1]
        assert 0 <= extreme_processed.loc[0, 'min_PT_final_state_tracks'] <= 1
        assert 0 <= extreme_processed.loc[1, 'min_FS_OWNPVIPCHI2'] <= 1
        
        # The extreme values should be clipped to 1 (upper bound)
        assert extreme_processed.loc[0, 'min_PT_final_state_tracks'] == 1
        assert extreme_processed.loc[1, 'min_FS_OWNPVIPCHI2'] == 1
    
    def test_normalization(self, preprocessor, sample_data):
        """Test that normalization correctly maps to [0,1]"""
        # Apply fit_transform
        preprocessor.fit(sample_data)
        processed_df = preprocessor.transform(sample_data)
        
        # Check that all features are scaled to [0,1]
        for feature in preprocessor.transformations:
            assert processed_df[feature].min() >= 0
            assert processed_df[feature].max() <= 1
            
            # Min value should be close to 0, max value close to 1
            assert np.isclose(processed_df[feature].min(), 0, atol=1e-5)
            assert np.isclose(processed_df[feature].max(), 1, atol=1e-5)
    
    def test_save_load(self, preprocessor, sample_data, tmp_path):
        """Test that saving and loading preserves transformation parameters"""
        # Fit the preprocessor
        preprocessor.fit(sample_data)
        
        # Save to file
        save_path = os.path.join(tmp_path, "preprocessor.pkl")
        preprocessor.save(save_path)
        
        # Load the preprocessor
        loaded = DataPreprocessor.load(save_path)
        
        # Check that configurations match
        assert loaded.gev_vars == preprocessor.gev_vars
        assert loaded.log_vars == preprocessor.log_vars
        assert loaded.unchanged_vars == preprocessor.unchanged_vars
        assert loaded.clip_quantiles == preprocessor.clip_quantiles
        assert loaded.normalize == preprocessor.normalize
        
        # Check that feature stats are preserved
        for key in preprocessor.feature_stats:
            assert key in loaded.feature_stats
            if isinstance(preprocessor.feature_stats[key], float):
                assert np.isclose(loaded.feature_stats[key], preprocessor.feature_stats[key])
        
        # Check that transformations give the same result
        processed1 = preprocessor.transform(sample_data)
        processed2 = loaded.transform(sample_data)
        
        # Compare transformed values
        for feature in preprocessor.transformations:
            np.testing.assert_array_almost_equal(
                processed1[feature].values, 
                processed2[feature].values
            )

if __name__ == "__main__":
    pytest.main(["-xvs"])