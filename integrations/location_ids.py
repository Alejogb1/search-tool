
import argparse
from typing import List
import sys

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from google.ads.googleads.v21.common.types.criteria import (
    KeywordInfo,
)
from google.ads.googleads.v21.resources.types.campaign_criterion import (
    CampaignCriterion,
)
from google.ads.googleads.v21.services.services.campaign_service import (
    CampaignServiceClient,
)
from google.ads.googleads.v21.services.services.campaign_criterion_service import (
    CampaignCriterionServiceClient,
)
from google.ads.googleads.v21.services.services.geo_target_constant_service import (
    GeoTargetConstantServiceClient,
)
from google.ads.googleads.v21.services.types.campaign_criterion_service import (
    CampaignCriterionOperation,
    MutateCampaignCriteriaResponse,
)

def create_location_op(
    client: GoogleAdsClient,
    customer_id: str,
    campaign_id: str,
    location_id: str,
) -> CampaignCriterionOperation:
    campaign_service: CampaignServiceClient = client.get_service(
        "CampaignService"
    )
    geo_target_constant_service: GeoTargetConstantServiceClient = (
        client.get_service("GeoTargetConstantService")
    )

    # Create the campaign criterion.
    campaign_criterion_operation: CampaignCriterionOperation = client.get_type(
        "CampaignCriterionOperation"
    )
    campaign_criterion: CampaignCriterion = campaign_criterion_operation.create
    campaign_criterion.campaign = campaign_service.campaign_path(
        customer_id, campaign_id
    )

    # Besides using location_id, you can also search by location names from
    # GeoTargetConstantService.suggest_geo_target_constants() and directly
    # apply GeoTargetConstant.resource_name here. An example can be found
    # in get_geo_target_constant_by_names.py.
    campaign_criterion.location.geo_target_constant = (
        geo_target_constant_service.geo_target_constant_path(location_id)
    )

    return campaign_criterion_operation