#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This example generates keyword ideas from a list of seed keywords."""

import sys
from google.ads.googleads.client import GoogleAdsClient as AdsClient
from google.ads.googleads.errors import GoogleAdsException

class GoogleAdsClient:
    #   https://callmefred.com/country-and-language-codes-to-be-used-with-google-ads-api/
    _DEFAULT_LOCATION_IDS = ["1023191"]  # New York, NY
    _DEFAULT_LOCATION_IDS = ["2344"]  # Berlin, Germany
    #_DEFAULT_LOCATION_IDS = ["1023191"]  # New York, NY
    #_DEFAULT_LOCATION_IDS = ["20010"]  # Buenos Aires, BA

    # Argentina
    #_DEFAULT_LOCATION_IDS = ["2032"] 

    # Belize
    #_DEFAULT_LOCATION_IDS = ["2084"]

    # Bolivia
    #_DEFAULT_LOCATION_IDS = ["2068"]

    # Brazil
    #_DEFAULT_LOCATION_IDS = ["2076"]

    # Chile
   # _DEFAULT_LOCATION_IDS = ["2152"]

    # Colombia
   # _DEFAULT_LOCATION_IDS = ["2170"]

    # Costa Rica
 #   _DEFAULT_LOCATION_IDS = ["2188"]

    # Dominica
  #  _DEFAULT_LOCATION_IDS = ["2212"]

    # Dominican Republic
  #  _DEFAULT_LOCATION_IDS = ["2214"]

    # Ecuador
  #  _DEFAULT_LOCATION_IDS = ["2218"]

    # El Salvador
  #  _DEFAULT_LOCATION_IDS = ["2222"]

    # Grenada
  #  _DEFAULT_LOCATION_IDS = ["2308"]

    # Guatemala
    #_DEFAULT_LOCATION_IDS = ["2320"]

    # Guyana
  #  _DEFAULT_LOCATION_IDS = ["2328"]

    # Haiti
   # _DEFAULT_LOCATION_IDS = ["2332"]

    # Honduras
   # _DEFAULT_LOCATION_IDS = ["2340"]

    # Jamaica
  #  _DEFAULT_LOCATION_IDS = ["2388"]

    # Mexico
    #_DEFAULT_LOCATION_IDS = ["2484"]

    # Nicaragua
  #  _DEFAULT_LOCATION_IDS = ["2558"]

    # Panama
   # _DEFAULT_LOCATION_IDS = ["2591"]

    # Paraguay
   # _DEFAULT_LOCATION_IDS = ["2600"]

    # Peru
   #_DEFAULT_LOCATION_IDS = ["2604"]

    # Saint Vincent and the Grenadines
    #_DEFAULT_LOCATION_IDS = ["2670"]

    # Suriname
  #  _DEFAULT_LOCATION_IDS = ["2740"]

    # Trinidad and Tobago
   # _DEFAULT_LOCATION_IDS = ["2780"]

    # Uruguay
   # _DEFAULT_LOCATION_IDS = ["2858"]

    # Venezuela
  #  _DEFAULT_LOCATION_IDS = ["2862"]

  #  _DEFAULT_LANGUAGE_ID = "1003"  # Spanish
    #_DEFAULT_LANGUAGE_ID = "1001"  # German

    _DEFAULT_LANGUAGE_ID = "1000"  # English
    def __init__(self, config_path='google-ads.yaml'):
        self.client = AdsClient.load_from_storage(config_path, version="v20")

    def generate_keyword_ideas(self, customer_id, keyword_texts, page_url=None, location_ids=None, language_id=None):
        location_ids = location_ids or self._DEFAULT_LOCATION_IDS
        language_id = language_id or self._DEFAULT_LANGUAGE_ID

        keyword_plan_idea_service = self.client.get_service("KeywordPlanIdeaService")
        keyword_plan_network = self.client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS
        
        location_rns = self._map_locations_ids_to_resource_names(location_ids)
        language_rn = self.client.get_service("GoogleAdsService").language_constant_path(language_id)

        if not (keyword_texts or page_url):
            raise ValueError("At least one of keywords or page URL is required.")

        request = self.client.get_type("GenerateKeywordIdeasRequest")
        request.customer_id = customer_id
        request.language = language_rn
        request.geo_target_constants = location_rns
        request.include_adult_keywords = False
        request.keyword_plan_network = keyword_plan_network

        if not keyword_texts and page_url:
            request.url_seed.url = page_url
        elif keyword_texts and not page_url:
            request.keyword_seed.keywords.extend(keyword_texts)
        elif keyword_texts and page_url:
            request.keyword_and_url_seed.url = page_url
            request.keyword_and_url_seed.keywords.extend(keyword_texts)

        try:
            keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(request=request)
            return [self._format_idea(idea) for idea in keyword_ideas]
        except GoogleAdsException as ex:
            self._handle_google_ads_exception(ex)
            return []

    def _format_idea(self, idea):
        competition_value = idea.keyword_idea_metrics.competition.name
        return {
            "text": idea.text,
            "avg_monthly_searches": idea.keyword_idea_metrics.avg_monthly_searches,
            "competition": competition_value
        }

    def _map_locations_ids_to_resource_names(self, location_ids):
        build_resource_name = self.client.get_service("GeoTargetConstantService").geo_target_constant_path
        return [build_resource_name(location_id) for location_id in location_ids]

    def _handle_google_ads_exception(self, ex):
        print(
            f'Request with ID "{ex.request_id}" failed with status '
            f'"{ex.error.code().name}" and includes the following errors:'
        )
        for error in ex.failure.errors:
            print(f'\tError with message "{error.message}".')
            if error.location:
                for field_path_element in error.location.field_path_elements:
                    print(f"\t\tOn field: {field_path_element.field_name}")
        sys.exit(1)

    def create_keyword_plan(self, customer_id, plan_name, keywords, url):
        """Create a keyword plan campaign using the latest API version"""
        try:
            # Initialize services
            keyword_plan_service = self.client.get_service("KeywordPlanService")
            campaign_service = self.client.get_service("KeywordPlanCampaignService")
            ad_group_service = self.client.get_service("KeywordPlanAdGroupService")
            keyword_service = self.client.get_service("KeywordPlanKeywordService")

            # Create keyword plan
            keyword_plan = self.client.get_type("KeywordPlan")
            keyword_plan.name = plan_name
            keyword_plan_operation = self.client.get_type("KeywordPlanOperation")
            keyword_plan_operation.create.CopyFrom(keyword_plan)
            
            plan_response = keyword_plan_service.mutate_keyword_plans(
                customer_id=customer_id,
                operations=[keyword_plan_operation]
            )
            plan_resource_name = plan_response.results[0].resource_name

            # Create campaign
            campaign = self.client.get_type("KeywordPlanCampaign")
            campaign.name = f"{plan_name} Campaign"
            campaign.cpc_bid_micros = 1000000  # 1 USD
            campaign.keyword_plan_network = self.client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
            campaign.geo_targets.extend(self._get_geo_targets())
            campaign.language_constants.append(self._get_language_constant())
            
            campaign_operation = self.client.get_type("KeywordPlanCampaignOperation")
            campaign_operation.create.CopyFrom(campaign)
            
            campaign_response = campaign_service.mutate_keyword_plan_campaigns(
                customer_id=customer_id,
                operations=[campaign_operation]
            )
            campaign_resource_name = campaign_response.results[0].resource_name

            # Create ad group
            ad_group = self.client.get_type("KeywordPlanAdGroup")
            ad_group.name = f"{plan_name} Ad Group"
            ad_group.cpc_bid_micros = 1000000  # 1 USD
            
            ad_group_operation = self.client.get_type("KeywordPlanAdGroupOperation")
            ad_group_operation.create.CopyFrom(ad_group)
            
            ad_group_response = ad_group_service.mutate_keyword_plan_ad_groups(
                customer_id=customer_id,
                operations=[ad_group_operation]
            )
            ad_group_resource_name = ad_group_response.results[0].resource_name

            # Add keywords
            keyword_operations = []
            for keyword_text in keywords:
                keyword = self.client.get_type("KeywordPlanKeyword")
                keyword.text = keyword_text
                keyword.match_type = self.client.enums.KeywordMatchType.BROAD
                keyword.cpc_bid_micros = 1000000
                keyword_operation = self.client.get_type("KeywordPlanKeywordOperation")
                keyword_operation.create.CopyFrom(keyword)
                keyword_operations.append(keyword_operation)

            keyword_service.mutate_keyword_plan_keywords(
                customer_id=customer_id,
                operations=keyword_operations
            )

            return {
                "plan_id": plan_resource_name,
                "campaign_id": campaign_resource_name,
                "ad_group_id": ad_group_resource_name
            }

        except GoogleAdsException as ex:
            self._handle_google_ads_exception(ex)
            return None

    def _get_geo_targets(self):
        geo_target_service = self.client.get_service("GeoTargetConstantService")
        return [
            geo_target_service.geo_target_constant_path(location_id)
            for location_id in self._DEFAULT_LOCATION_IDS
        ]

    def _get_language_constant(self):
        return self.client.get_service("GoogleAdsService").language_constant_path(
            self._DEFAULT_LANGUAGE_ID
        )
