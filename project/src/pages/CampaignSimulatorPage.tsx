import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { 
  Search, DollarSign, TrendingUp, MapPin, 
  Globe, Building2, Building, Calculator 
} from 'lucide-react';
import { CampaignSimulation, Location } from '../types/campaign';

const [formData, setFormData] = useState({
    region: 'North America', // Valor inicial de exemplo
    content: 'Image',       // Valor inicial de exemplo
    age: '25-34',         // Valor inicial de exemplo
    gender: 'Female',     // Valor inicial de exemplo
    platform: 'Instagram',  // Valor inicial de exemplo
    budget: 1000,           // Mantém o budget
    product_tier: 'Low Ticket' // Valor inicial de exemplo
  });
  
  const [showResults, setShowResults] = useState(false);

  const { data: simulation, isLoading, error, refetch } = useQuery({
    queryKey: ['campaignSimulation', formData],
queryFn: async () => {
      // Faz a chamada de API real para o seu servidor Python local
      const response = await axios.post(
        'http://127.0.0.1:8000/api/simulate', 
        formData // Envia o estado do formulário
      );
      
      // A API de RL retorna apenas a recomendação.
      // Precisamos "encaixar" isso na sua interface 'CampaignSimulation'
      const rlData = response.data; // ex: { recommended_price: 29.99, ... }

      // Como a API de RL não retorna dados de localização, vamos usar mocks
      // ou apenas focar em mostrar a recomendação de preço.
      // Por enquanto, vamos retornar um objeto simples:
      
      // ATENÇÃO: A sua API de Python (main.py) retorna um formato simples:
      // { "recommended_price": ..., "estimated_roi": ... }
      // A sua página espera um objeto 'CampaignSimulation'
      // Vamos precisar adaptar a resposta:
      
      const recommendation: PricingRecommendation = {
        type: formData.product_tier === 'Mid Ticket' ? 'subscription' : 'fixed',
        amount: rlData.recommended_price,
        estimatedRevenue: (formData.budget * (rlData.estimated_roi / 100)) + formData.budget,
        roi: rlData.estimated_roi,
        coverage: 0.0, // Placeholder
        locations: [formData.region] // Usa a região do input
      };
      
      const simulationResult: CampaignSimulation = {
        keyword: 'Simulação de RL', // Placeholder
        budget: formData.budget,
        targetRevenue: 0.0, // Placeholder
        pricingModel: 'fixed', // Placeholder
        countries: [], // API de RL não retorna isso
        states: [],    // API de RL não retorna isso
        cities: [],    // API de RL não retorna isso
        recommendations: [recommendation] // Insere a recomendação da API
      };
      
      return simulationResult;
    },

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setShowResults(true);
    refetch();
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  return (
    <div className="pb-16 lg:pb-0">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Campaign Simulator
          </h1>

          {/* Input Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Keyword Input */}
              <div>
                <label htmlFor="keyword" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Target Keyword
                </label>
                <div className="mt-1 relative rounded-md shadow-sm">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Search className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    type="text"
                    id="keyword"
                    value={formData.keyword}
                    onChange={(e) => setFormData({ ...formData, keyword: e.target.value })}
                    className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="e.g., digital marketing"
                    required
                  />
                </div>
              </div>

              {/* Monthly Budget */}
              <div>
                <label htmlFor="budget" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Monthly Budget
                </label>
                <div className="mt-1 relative rounded-md shadow-sm">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <DollarSign className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    type="number"
                    id="budget"
                    value={formData.budget}
                    onChange={(e) => setFormData({ ...formData, budget: Number(e.target.value) })}
                    className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="1000"
                    min="100"
                    required
                  />
                </div>
              </div>

              {/* Target Revenue */}
              <div>
                <label htmlFor="targetRevenue" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Target Monthly Revenue
                </label>
                <div className="mt-1 relative rounded-md shadow-sm">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <TrendingUp className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    type="number"
                    id="targetRevenue"
                    value={formData.targetRevenue}
                    onChange={(e) => setFormData({ ...formData, targetRevenue: Number(e.target.value) })}
                    className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="5000"
                    min="1000"
                    required
                  />
                </div>
              </div>

              {/* Pricing Model */}
              <div>
                <label htmlFor="pricingModel" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Pricing Model
                </label>
                <div className="mt-1">
                  <select
                    id="pricingModel"
                    value={formData.pricingModel}
                    onChange={(e) => setFormData({ ...formData, pricingModel: e.target.value as 'subscription' | 'fixed' })}
                    className="block w-full pl-3 pr-10 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    <option value="subscription">Monthly Subscription</option>
                    <option value="fixed">Fixed Price</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="flex justify-end">
              <button
                type="submit"
                disabled={isLoading}
                className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <span className="flex items-center">
                    <Calculator className="animate-spin -ml-1 mr-2 h-4 w-4" />
                    Simulating...
                  </span>
                ) : (
                  <span className="flex items-center">
                    <Calculator className="mr-2 h-4 w-4" />
                    Run Simulation
                  </span>
                )}
              </button>
            </div>
          </form>

          {/* Results */}
          {showResults && !isLoading && simulation && (
            <div className="mt-8 space-y-6">
              <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Top Locations by Search Volume and CPC
                </h2>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* Countries */}
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center mb-4">
                      <Globe className="h-5 w-5 text-primary-500 mr-2" />
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                        Top Countries
                      </h3>
                    </div>
                    <ul className="space-y-3">
                      {simulation.countries.slice(0, 5).map((location, index) => (
                        <li key={index} className="flex items-center justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-300">{location.name}</span>
                          <span className="text-gray-900 dark:text-white font-medium">
                            {formatCurrency(location.cpc)} CPC
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* States */}
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center mb-4">
                      <Building2 className="h-5 w-5 text-secondary-500 mr-2" />
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                        Top States/Regions
                      </h3>
                    </div>
                    <ul className="space-y-3">
                      {simulation.states.slice(0, 5).map((location, index) => (
                        <li key={index} className="flex items-center justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-300">{location.name}</span>
                          <span className="text-gray-900 dark:text-white font-medium">
                            {formatCurrency(location.cpc)} CPC
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Cities */}
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center mb-4">
                      <Building className="h-5 w-5 text-accent-500 mr-2" />
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                        Top Cities
                      </h3>
                    </div>
                    <ul className="space-y-3">
                      {simulation.cities.slice(0, 5).map((location, index) => (
                        <li key={index} className="flex items-center justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-300">{location.name}</span>
                          <span className="text-gray-900 dark:text-white font-medium">
                            {formatCurrency(location.cpc)} CPC
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>

              {/* Pricing Recommendations */}
              <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Pricing Recommendations
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {simulation.recommendations.map((rec, index) => (
                    <div 
                      key={index}
                      className="bg-white dark:bg-gray-700 rounded-lg shadow-sm border border-gray-200 dark:border-gray-600 p-6"
                    >
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                          Option {index + 1}
                        </h3>
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 dark:bg-primary-900/30 text-primary-800 dark:text-primary-200">
                          {rec.type === 'subscription' ? 'Monthly' : 'One-time'}
                        </span>
                      </div>

                      <div className="space-y-4">
                        <div>
                          <p className="text-3xl font-bold text-gray-900 dark:text-white">
                            {formatCurrency(rec.amount)}
                          </p>
                          <p className="text-sm text-gray-500 dark:text-gray-400">
                            {rec.type === 'subscription' ? 'per month' : 'fixed price'}
                          </p>
                        </div>

                        <ul className="space-y-2">
                          <li className="flex items-center justify-between text-sm">
                            <span className="text-gray-500 dark:text-gray-400">Est. Revenue</span>
                            <span className="text-gray-900 dark:text-white font-medium">
                              {formatCurrency(rec.estimatedRevenue)}
                            </span>
                          </li>
                          <li className="flex items-center justify-between text-sm">
                            <span className="text-gray-500 dark:text-gray-400">ROI</span>
                            <span className="text-gray-900 dark:text-white font-medium">
                              {rec.roi}%
                            </span>
                          </li>
                          <li className="flex items-center justify-between text-sm">
                            <span className="text-gray-500 dark:text-gray-400">Market Coverage</span>
                            <span className="text-gray-900 dark:text-white font-medium">
                              {(rec.coverage * 100).toFixed(1)}%
                            </span>
                          </li>
                        </ul>

                        <div className="pt-4 border-t border-gray-200 dark:border-gray-600">
                          <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Target Locations:
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {rec.locations.map((location, idx) => (
                              <span 
                                key={idx}
                                className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-600 text-gray-800 dark:text-gray-200"
                              >
                                <MapPin className="h-3 w-3 mr-1" />
                                {location}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="mt-4 bg-accent-50 dark:bg-accent-900/10 border-l-4 border-accent-500 p-4 rounded">
              <p className="text-sm text-accent-700 dark:text-accent-400">
                An error occurred while simulating the campaign. Please try again.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CampaignSimulatorPage;
