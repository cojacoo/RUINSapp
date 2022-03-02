from ruins import components


def test_default():
    """Test default behavior"""
    assert components.topic_selector(['a', 'b', 'c'], no_cache=True) == 'a'


def test_current_preset():
    """Test with current topic set"""
    topic = components.topic_selector(['a', 'b', 'c'], current_topic='b', no_cache=True)

    assert topic == 'b'


def test_no_force_render():
    """Test with rendering disabled"""
    # default
    assert components.topic_selector(['a', 'b', 'c'], force_topic_select=False, no_cache=True) == 'a'

    # with current topic set
    assert components.topic_selector(['a', 'b', 'c'], current_topic='b', force_topic_select=False, no_cache=True) == 'b'
    