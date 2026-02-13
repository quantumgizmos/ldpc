"""
Tests for check_to_bit_clip_value parameter in hybrid decoders.
Tests cover BpOsdDecoder, BpLsdDecoder, and BeliefFindDecoder.
"""

import numpy as np
import pytest
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.belief_find_decoder import BeliefFindDecoder


class TestBpOsdDecoderClipValue:
    """Test check_to_bit_clip_value for BpOsdDecoder."""

    @pytest.fixture
    def simple_pcm(self):
        """Create a simple parity check matrix for testing."""
        return np.array([
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1]
        ], dtype=np.uint8)

    def test_default_check_to_bit_clip_value(self, simple_pcm):
        """Test BpOsdDecoder has default check_to_bit_clip_value of 1000."""
        decoder = BpOsdDecoder(simple_pcm, error_rate=0.1)
        assert decoder.check_to_bit_clip_value == 1000

    def test_custom_check_to_bit_clip_value_at_init(self, simple_pcm):
        """Test setting custom check_to_bit_clip_value during init."""
        custom_value = 50.0
        decoder = BpOsdDecoder(
            simple_pcm,
            error_rate=0.1,
            check_to_bit_clip_value=custom_value
        )
        assert decoder.check_to_bit_clip_value == custom_value

    def test_check_to_bit_clip_value_property_setter(self, simple_pcm):
        """Test setting check_to_bit_clip_value via property."""
        decoder = BpOsdDecoder(simple_pcm, error_rate=0.1)
        new_value = 75.0
        decoder.check_to_bit_clip_value = new_value
        assert decoder.check_to_bit_clip_value == new_value

    def test_decode_with_custom_clip_value(self, simple_pcm):
        """Test decoding works with custom clip value."""
        decoder = BpOsdDecoder(
            simple_pcm,
            error_rate=0.1,
            max_iter=5,
            check_to_bit_clip_value=30.0
        )
        syndrome = np.array([0, 0, 0], dtype=np.uint8)
        result = decoder.decode(syndrome)
        assert isinstance(result, np.ndarray)
        assert len(result) == simple_pcm.shape[1]


class TestBpLsdDecoderClipValue:
    """Test check_to_bit_clip_value for BpLsdDecoder."""

    @pytest.fixture
    def simple_pcm(self):
        """Create a simple parity check matrix for testing."""
        return np.array([
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1]
        ], dtype=np.uint8)

    def test_default_check_to_bit_clip_value(self, simple_pcm):
        """Test BpLsdDecoder has default check_to_bit_clip_value of 1000."""
        decoder = BpLsdDecoder(simple_pcm, error_rate=0.1)
        assert decoder.check_to_bit_clip_value == 1000

    def test_custom_check_to_bit_clip_value_at_init(self, simple_pcm):
        """Test setting custom check_to_bit_clip_value during init."""
        custom_value = 60.0
        decoder = BpLsdDecoder(
            simple_pcm,
            error_rate=0.1,
            check_to_bit_clip_value=custom_value
        )
        assert decoder.check_to_bit_clip_value == custom_value

    def test_check_to_bit_clip_value_property_setter(self, simple_pcm):
        """Test setting check_to_bit_clip_value via property."""
        decoder = BpLsdDecoder(simple_pcm, error_rate=0.1)
        new_value = 85.0
        decoder.check_to_bit_clip_value = new_value
        assert decoder.check_to_bit_clip_value == new_value

    def test_decode_with_custom_clip_value(self, simple_pcm):
        """Test decoding works with custom clip value."""
        decoder = BpLsdDecoder(
            simple_pcm,
            error_rate=0.1,
            max_iter=5,
            bits_per_step=2,
            check_to_bit_clip_value=40.0
        )
        syndrome = np.array([0, 0, 0], dtype=np.uint8)
        result = decoder.decode(syndrome)
        assert isinstance(result, np.ndarray)
        assert len(result) == simple_pcm.shape[1]

    def test_with_product_sum_method(self, simple_pcm):
        """Test with PRODUCT_SUM method which uses clipping."""
        decoder = BpLsdDecoder(
            simple_pcm,
            error_rate=0.1,
            max_iter=5,
            bp_method='product_sum',
            check_to_bit_clip_value=25.0
        )
        assert decoder.check_to_bit_clip_value == 25.0
        syndrome = np.array([0, 0, 0], dtype=np.uint8)
        result = decoder.decode(syndrome)
        assert isinstance(result, np.ndarray)


class TestBeliefFindDecoderClipValue:
    """Test check_to_bit_clip_value for BeliefFindDecoder."""

    @pytest.fixture
    def simple_pcm(self):
        """Create a simple parity check matrix for testing."""
        return np.array([
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1]
        ], dtype=np.uint8)

    def test_default_check_to_bit_clip_value(self, simple_pcm):
        """Test BeliefFindDecoder has default check_to_bit_clip_value of 1000."""
        decoder = BeliefFindDecoder(simple_pcm, error_rate=0.1)
        assert decoder.check_to_bit_clip_value == 1000

    def test_custom_check_to_bit_clip_value_at_init(self, simple_pcm):
        """Test setting custom check_to_bit_clip_value during init."""
        custom_value = 45.0
        decoder = BeliefFindDecoder(
            simple_pcm,
            error_rate=0.1,
            check_to_bit_clip_value=custom_value
        )
        assert decoder.check_to_bit_clip_value == custom_value

    def test_check_to_bit_clip_value_property_setter(self, simple_pcm):
        """Test setting check_to_bit_clip_value via property."""
        decoder = BeliefFindDecoder(simple_pcm, error_rate=0.1)
        new_value = 65.0
        decoder.check_to_bit_clip_value = new_value
        assert decoder.check_to_bit_clip_value == new_value

    def test_decode_with_custom_clip_value(self, simple_pcm):
        """Test decoding works with custom clip value."""
        decoder = BeliefFindDecoder(
            simple_pcm,
            error_rate=0.1,
            max_iter=5,
            bits_per_step=1,
            uf_method='inversion',
            check_to_bit_clip_value=35.0
        )
        syndrome = np.array([0, 0, 0], dtype=np.uint8)
        result = decoder.decode(syndrome)
        assert isinstance(result, np.ndarray)
        assert len(result) == simple_pcm.shape[1]

    def test_with_product_sum_method(self, simple_pcm):
        """Test with PRODUCT_SUM method which uses clipping."""
        decoder = BeliefFindDecoder(
            simple_pcm,
            error_rate=0.1,
            max_iter=5,
            bp_method='product_sum',
            uf_method='inversion',
            check_to_bit_clip_value=20.0
        )
        assert decoder.check_to_bit_clip_value == 20.0
        syndrome = np.array([0, 0, 0], dtype=np.uint8)
        result = decoder.decode(syndrome)
        assert isinstance(result, np.ndarray)


class TestAllDecodersConsistency:
    """Test that all decoders handle check_to_bit_clip_value consistently."""

    @pytest.fixture
    def simple_pcm(self):
        """Create a simple parity check matrix for testing."""
        return np.array([
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1]
        ], dtype=np.uint8)

    def test_all_decoders_accept_clip_value(self, simple_pcm):
        """Test all decoders accept check_to_bit_clip_value parameter."""
        clip_value = 55.0
        
        bp_osd = BpOsdDecoder(simple_pcm, error_rate=0.1, check_to_bit_clip_value=clip_value)
        bp_lsd = BpLsdDecoder(simple_pcm, error_rate=0.1, check_to_bit_clip_value=clip_value)
        bf_uf = BeliefFindDecoder(simple_pcm, error_rate=0.1, check_to_bit_clip_value=clip_value)
        
        assert bp_osd.check_to_bit_clip_value == clip_value
        assert bp_lsd.check_to_bit_clip_value == clip_value
        assert bf_uf.check_to_bit_clip_value == clip_value

    def test_all_decoders_accept_type_validation(self, simple_pcm):
        """Test that all decoders reject invalid types."""
        bp_osd = BpOsdDecoder(simple_pcm, error_rate=0.1)
        bp_lsd = BpLsdDecoder(simple_pcm, error_rate=0.1)
        bf_uf = BeliefFindDecoder(simple_pcm, error_rate=0.1)
        
        for decoder in [bp_osd, bp_lsd, bf_uf]:
            with pytest.raises(TypeError):
                decoder.check_to_bit_clip_value = "invalid"

    def test_all_decoders_accept_int_value(self, simple_pcm):
        """Test that all decoders accept integer clip values."""
        clip_value = 50  # integer
        
        decoders = [
            BpOsdDecoder(simple_pcm, error_rate=0.1, check_to_bit_clip_value=clip_value),
            BpLsdDecoder(simple_pcm, error_rate=0.1, check_to_bit_clip_value=clip_value),
            BeliefFindDecoder(simple_pcm, error_rate=0.1, check_to_bit_clip_value=clip_value)
        ]
        
        for decoder in decoders:
            assert decoder.check_to_bit_clip_value == clip_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
