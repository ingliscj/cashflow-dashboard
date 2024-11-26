import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Calendar, DollarSign, AlertTriangle, TrendingUp } from 'lucide-react';

// Format large numbers with commas
const formatNumber = (num) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'AED'
  }).format(num);
};

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/cashflow')
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to load data');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-xl">Loading dashboard data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen text-red-500">
        <AlertTriangle className="mr-2" /> {error}
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <h1 className="text-3xl font-bold mb-8">Cashflow Dashboard</h1>
      
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center mb-2">
            <DollarSign className="text-green-500 mr-2" />
            <h3 className="text-lg font-semibold">Total Upcoming Payments</h3>
          </div>
          <p className="text-2xl font-bold">{formatNumber(data?.total_upcoming_payments || 0)}</p>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center mb-2">
            <Calendar className="text-blue-500 mr-2" />
            <h3 className="text-lg font-semibold">Period</h3>
          </div>
          <p className="text-2xl font-bold">{data?.period || 'N/A'}</p>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center mb-2">
            <TrendingUp className="text-purple-500 mr-2" />
            <h3 className="text-lg font-semibold">Total Invoices</h3>
          </div>
          <p className="text-2xl font-bold">{data?.summary?.total_invoices || 0}</p>
        </div>
      </div>

      {/* Payments by Entity Chart */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mb-8">
        <h2 className="text-xl font-semibold mb-4">Payments by Entity</h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={Object.entries(data?.payments_by_entity || {}).map(([entity, amount]) => ({
              entity,
              amount
            }))}>
              <XAxis dataKey="entity" />
              <YAxis tickFormatter={(value) => formatNumber(value)} />
              <Tooltip 
                formatter={(value) => formatNumber(value)}
                labelFormatter={(label) => `Entity: ${label}`}
              />
              <Bar dataKey="amount" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recommendations Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <h2 className="text-xl font-semibold p-6 border-b">Payment Recommendations</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Entity</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Amount</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Recommendation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {data?.daily_recommendations?.map((rec, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">{rec.date}</td>
                  <td className="px-6 py-4">{rec.entity}</td>
                  <td className="px-6 py-4">{formatNumber(rec.amount)}</td>
                  <td className="px-6 py-4">{rec.recommendation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
