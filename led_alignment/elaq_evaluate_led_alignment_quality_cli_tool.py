import textwrap
from elaq_evaluate_led_alignment_quality import (
     elaq_evaluate_led_alignment_quality
)

import argparse

def elaq_evaluate_led_alignment_quality_cli_tool():

    usage_str = textwrap.dedent(
        """\
        See possible ad_ids via:
        bat ~/r/nba_ads/slgame1.json5

        Then do something like this:

        elaq_evaluate_led_alignment_quality 2024-SummerLeague_Courtside_2520x126_TM_STILL
        """
    )

    argparser = argparse.ArgumentParser(
        description="elaq is for evaluating the quality of the alignments of an LED ad with video frames.",
        usage=usage_str,
    )
    
    argparser.add_argument(
        "ad_id",
        type=str,
    )
    opt = argparser.parse_args()
    ad_id = opt.ad_id

    elaq_evaluate_led_alignment_quality(  
        ad_id=ad_id,
    )
