export interface Location {
  name: string;
  searchVolume: number;
  cpc: number;
  competition: number;
}

export interface CampaignSimulation {
  keyword: string;
  budget: number;
  targetRevenue: number;
  pricingModel: 'subscription' | 'fixed';
  countries: Location[];
  states: Location[];
  cities: Location[];
  recommendations: PricingRecommendation[];
}

export interface PricingRecommendation {
  type: 'subscription' | 'fixed';
  amount: number;
  estimatedRevenue: number;
  roi: number;
  coverage: number;
  locations: string[];
}