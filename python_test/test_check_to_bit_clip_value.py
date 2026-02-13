"""
Tests for the check_to_bit_clip_value parameter in the BP decoder.
"""

import numpy as np
import pytest
from ldpc.bp_decoder import BpDecoder


class TestCheckToBitClipValue:
    """Test suite for check_to_bit_clip_value parameter."""

    @pytest.fixture
    def simple_pcm(self):
        """Create a simple parity check matrix for testing."""
        # Simple repetition code PCM: H = [1 1 0; 0 1 1]
        return np.array([
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1]
        ], dtype=np.uint8)

    def test_default_check_to_bit_clip_value(self, simple_pcm):
        """Test that default check_to_bit_clip_value is 1000."""
        error_rate = 0.1
        decoder = BpDecoder(simple_pcm, error_rate=error_rate)
        
        assert decoder.check_to_bit_clip_value == 1000, \
            f"Default check_to_bit_clip_value should be 1000, got {decoder.check_to_bit_clip_value}"

    def test_custom_check_to_bit_clip_value_at_init(self, simple_pcm):
        """Test setting custom check_to_bit_clip_value during initialization."""
        error_rate = 0.1
        custom_clip_value = 50.0
        decoder = BpDecoder(
            simple_pcm, 
            error_rate=error_rate,
            check_to_bit_clip_value=custom_clip_value
        )
        
        assert decoder.check_to_bit_clip_value == custom_clip_value, \
            f"check_to_bit_clip_value should be {custom_clip_value}, got {decoder.check_to_bit_clip_value}"

    def test_check_to_bit_clip_value_property_setter(self, simple_pcm):
        """Test setting check_to_bit_clip_value via property setter."""
        error_rate = 0.1
        decoder = BpDecoder(simple_pcm, error_rate=error_rate)
        
        new_clip_value = 75.0
        decoder.check_to_bit_clip_value = new_clip_value
        
        assert decoder.check_to_bit_clip_value == new_clip_value, \
            f"check_to_bit_clip_value should be {new_clip_value}, got {decoder.check_to_bit_clip_value}"

    def test_check_to_bit_clip_value_type_validation(self, simple_pcm):
        """Test that invalid types are rejected for check_to_bit_clip_value."""
        error_rate = 0.1
        decoder = BpDecoder(simple_pcm, error_rate=error_rate)
        
        with pytest.raises(TypeError):
            decoder.check_to_bit_clip_value = "invalid_string"
        
        with pytest.raises(TypeError):
            decoder.check_to_bit_clip_value = [1, 2, 3]

    def test_check_to_bit_clip_value_with_product_sum(self, simple_pcm):
        """Test check_to_bit_clip_value with PRODUCT_SUM method."""
        error_rate = 0.01  # Low error rate to generate larger messages
        decoder = BpDecoder(
            simple_pcm,
            error_rate=error_rate,
            max_iter=5,
            bp_method='product_sum',
            check_to_bit_clip_value=10.0
        )
        
        assert decoder.check_to_bit_clip_value == 10.0
        
        # Decode a syndrome
        syndrome = np.array([0, 0, 0], dtype=np.uint8)
        result = decoder.decode(syndrome)
        
        # Should return a valid decoding
        assert isinstance(result, np.ndarray)
        assert len(result) == simple_pcm.shape[1]

    def test_check_to_bit_clip_value_with_minimum_sum(self, simple_pcm):
        """Test check_to_bit_clip_value with MINIMUM_SUM method (should not affect)."""
        error_rate = 0.1
        clip_value = 20.0
        decoder = BpDecoder(
            simple_pcm,
            error_rate=error_rate,
            max_iter=5,
            bp_method='minimum_sum',
            check_to_bit_clip_value=clip_value
        )
        
        assert decoder.check_to_bit_clip_value == clip_value
        
        # MINIMUM_SUM doesn't use the clipping, but it should be settable
        syndrome = np.array([0, 0, 0], dtype=np.uint8)
        result = decoder.decode(syndrome)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == simple_pcm.shape[1]

    def test_check_to_bit_clip_value_small_value(self, simple_pcm):
        """Test with a very small clipping value."""
        error_rate = 0.1
        decoder = BpDecoder(
            simple_pcm,
            error_rate=error_rate,
            max_iter=5,
            bp_method='product_sum',
            check_to_bit_clip_value=0.1
        )
        
        assert decoder.check_to_bit_clip_value == 0.1
        
        syndrome = np.array([0, 0, 0], dtype=np.uint8)
        result = decoder.decode(syndrome)
        
        assert isinstance(result, np.ndarray)

    def test_check_to_bit_clip_value_large_value(self, simple_pcm):
        """Test with a very large clipping value."""
        error_rate = 0.1
        decoder = BpDecoder(
            simple_pcm,
            error_rate=error_rate,
            max_iter=5,
            bp_method='product_sum',
            check_to_bit_clip_value=10000.0
        )
        
        assert decoder.check_to_bit_clip_value == 10000.0
        
        syndrome = np.array([0, 0, 0], dtype=np.uint8)
        result = decoder.decode(syndrome)
        
        assert isinstance(result, np.ndarray)

    def test_check_to_bit_clip_value_change_between_decodes(self, simple_pcm):
        """Test changing check_to_bit_clip_value between successive decodes."""
        error_rate = 0.1
        decoder = BpDecoder(simple_pcm, error_rate=error_rate, max_iter=5, bp_method='product_sum')
        
        syndrome = np.array([0, 0, 0], dtype=np.uint8)
        
        # First decode with default value
        result1 = decoder.decode(syndrome)
        assert decoder.check_to_bit_clip_value == 1000
        
        # Change the value
        decoder.check_to_bit_clip_value = 30.0
        
        # Second decode with new value
        result2 = decoder.decode(syndrome)
        assert decoder.check_to_bit_clip_value == 30.0
        
        # Both should be valid results
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)

    def test_check_to_bit_clip_value_accepts_int(self, simple_pcm):
        """Test that check_to_bit_clip_value accepts integer values."""
        error_rate = 0.1
        int_clip_value = 50  # Integer instead of float
        decoder = BpDecoder(
            simple_pcm,
            error_rate=error_rate,
            check_to_bit_clip_value=int_clip_value
        )
        
        # Should be accepted and usable
        assert decoder.check_to_bit_clip_value == int_clip_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
