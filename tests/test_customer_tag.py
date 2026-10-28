"""cognocient_customer maps to the X-Cost-Customer tag and never reaches the SDK call."""
from cognocient._tags import pop_tags


def test_customer_tag_is_popped_and_mapped():
    kwargs = {"model": "gpt-4o-mini", "cognocient_customer": "acme-corp", "cognocient_feature": "search"}
    assert pop_tags(kwargs) == {"tag_customer": "acme-corp", "tag_feature": "search"}
    assert kwargs == {"model": "gpt-4o-mini"}
