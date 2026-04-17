import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Activity, Users, Clock, AlertTriangle, TrendingUp, Calendar, RefreshCw } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const Dashboard = () => {
  const [predictions, setPredictions] = useState({});
  const [allocations, setAllocations] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedDept, setSelectedDept] = useState('OPD');
  
  // Sample input state
  const [inputData, setInputData] = useState({
    hour: new Date().getHours(),
    day_of_week: new Date().getDay(),
    month: new Date().getMonth() + 1,
    queue_length_at_arrival: 8,
    total_counters_available: 4,
    arrival_rate: 2.0,
    system_utilization: 0.75,
    doctor_efficiency: 0.8,
    department: 'OPD'
  });

  const departments = ['OPD', 'Diagnostics', 'Pharmacy', 'Emergency'];
  
  const currentAllocations = {
    'OPD': 4,
    'Diagnostics': 2,
    'Pharmacy': 2,
    'Emergency': 3
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE}/metrics`);
      const data = await response.json();
      setMetrics(data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(inputData)
      });
      const data = await response.json();
      setPredictions(prev => ({ ...prev, [inputData.department]: data }));
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Error making prediction. Is the API server running?');
    }
    setLoading(false);
  };

  const handleAllocate = async () => {
    setLoading(true);
    try {
      // Build department predictions from current state
      const deptPredictions = {};
      departments.forEach(dept => {
        const pred = predictions[dept];
        deptPredictions[dept] = {
          wait_time: pred?.predicted_wait_time || 20,
          queue_length: inputData.queue_length_at_arrival,
          arrival_rate: inputData.arrival_rate,
          utilization: inputData.system_utilization
        };
      });

      const response = await fetch(`${API_BASE}/allocate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          department_predictions: deptPredictions,
          current_allocations: currentAllocations,
          total_staff: 15
        })
      });
      const data = await response.json();
      setAllocations(data);
    } catch (error) {
      console.error('Allocation error:', error);
      alert('Error getting allocation. Is the API server running?');
    }
    setLoading(false);
  };

  const getAlertColor = (level) => {
    const colors = {
      normal: '#10b981',
      warning: '#f59e0b',
      critical: '#ef4444'
    };
    return colors[level] || colors.normal;
  };

  const currentPrediction = predictions[selectedDept];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Animated background pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 2px 2px, white 1px, transparent 0)`,
          backgroundSize: '40px 40px'
        }} />
      </div>

      <div className="relative max-w-7xl mx-auto p-6 space-y-6">
        {/* Header */}
        <header className="text-center py-8 relative">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 via-blue-500/10 to-purple-500/10 blur-3xl" />
          <h1 className="relative text-5xl font-black bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent mb-3 tracking-tight" 
              style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            Hospital Queue Intelligence
          </h1>
          <p className="relative text-slate-400 text-lg" style={{ fontFamily: "'Inter', sans-serif" }}>
            AI-Powered Prediction & Smart Counter Allocation System
          </p>
        </header>

        {/* Quick Stats */}
        {metrics && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[
              { label: 'Model Accuracy', value: `${(metrics.test?.r2 * 100 || 0).toFixed(1)}%`, icon: TrendingUp, color: 'from-cyan-500 to-blue-500' },
              { label: 'Avg Error', value: `${metrics.test?.mae?.toFixed(1) || 0} min`, icon: Activity, color: 'from-blue-500 to-purple-500' },
              { label: 'RMSE', value: `${metrics.test?.rmse?.toFixed(1) || 0} min`, icon: Clock, color: 'from-purple-500 to-pink-500' },
              { label: 'Total Features', value: metrics.n_features || 0, icon: Calendar, color: 'from-pink-500 to-rose-500' }
            ].map((stat, i) => (
              <div key={i} className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-5 hover:border-slate-700/50 transition-all duration-300 group">
                <div className="flex items-center justify-between mb-3">
                  <stat.icon className="w-8 h-8 text-slate-600 group-hover:text-slate-500 transition-colors" />
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${stat.color} opacity-20 group-hover:opacity-30 transition-opacity`} />
                </div>
                <div className="text-3xl font-bold text-white mb-1">{stat.value}</div>
                <div className="text-sm text-slate-500">{stat.label}</div>
              </div>
            ))}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Panel */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6 space-y-4">
            <h2 className="text-2xl font-bold text-white flex items-center gap-2 mb-4">
              <Users className="w-6 h-6 text-cyan-400" />
              Queue Parameters
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">Department</label>
                <select 
                  value={inputData.department}
                  onChange={(e) => {
                    setInputData({...inputData, department: e.target.value});
                    setSelectedDept(e.target.value);
                  }}
                  className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-3 text-white focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 outline-none transition-all"
                >
                  {departments.map(dept => (
                    <option key={dept} value={dept}>{dept}</option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">Hour (0-23)</label>
                  <input 
                    type="number"
                    min="0"
                    max="23"
                    value={inputData.hour}
                    onChange={(e) => setInputData({...inputData, hour: parseInt(e.target.value)})}
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-3 text-white focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 outline-none transition-all"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">Queue Length</label>
                  <input 
                    type="number"
                    min="0"
                    value={inputData.queue_length_at_arrival}
                    onChange={(e) => setInputData({...inputData, queue_length_at_arrival: parseInt(e.target.value)})}
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-3 text-white focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 outline-none transition-all"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">Active Counters</label>
                  <input 
                    type="number"
                    min="1"
                    value={inputData.total_counters_available}
                    onChange={(e) => setInputData({...inputData, total_counters_available: parseInt(e.target.value)})}
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-3 text-white focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 outline-none transition-all"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">Arrival Rate</label>
                  <input 
                    type="number"
                    step="0.1"
                    min="0"
                    value={inputData.arrival_rate}
                    onChange={(e) => setInputData({...inputData, arrival_rate: parseFloat(e.target.value)})}
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-3 text-white focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 outline-none transition-all"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">Utilization (0-1)</label>
                  <input 
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={inputData.system_utilization}
                    onChange={(e) => setInputData({...inputData, system_utilization: parseFloat(e.target.value)})}
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-3 text-white focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 outline-none transition-all"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">Doctor Efficiency</label>
                  <input 
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={inputData.doctor_efficiency}
                    onChange={(e) => setInputData({...inputData, doctor_efficiency: parseFloat(e.target.value)})}
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-3 text-white focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 outline-none transition-all"
                  />
                </div>
              </div>

              <button 
                onClick={handlePredict}
                disabled={loading}
                className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white font-semibold py-4 px-6 rounded-xl transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-cyan-500/20"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    Processing...
                  </span>
                ) : (
                  'Predict Wait Time'
                )}
              </button>
            </div>
          </div>

          {/* Prediction Results */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6">
            <h2 className="text-2xl font-bold text-white flex items-center gap-2 mb-6">
              <Clock className="w-6 h-6 text-purple-400" />
              Prediction Results
            </h2>

            {currentPrediction ? (
              <div className="space-y-6">
                <div className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border border-cyan-500/20 rounded-2xl p-6 text-center">
                  <div className="text-sm text-slate-400 mb-2">Predicted Wait Time</div>
                  <div className="text-6xl font-black bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent mb-2">
                    {currentPrediction.predicted_wait_time}
                  </div>
                  <div className="text-2xl text-slate-300 font-semibold">minutes</div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-800/30 rounded-xl p-4 border border-slate-700/30">
                    <div className="text-xs text-slate-500 mb-1">Lower Bound</div>
                    <div className="text-2xl font-bold text-green-400">
                      {currentPrediction.confidence_interval.lower.toFixed(1)} min
                    </div>
                  </div>
                  <div className="bg-slate-800/30 rounded-xl p-4 border border-slate-700/30">
                    <div className="text-xs text-slate-500 mb-1">Upper Bound</div>
                    <div className="text-2xl font-bold text-orange-400">
                      {currentPrediction.confidence_interval.upper.toFixed(1)} min
                    </div>
                  </div>
                </div>

                <div className="bg-slate-800/20 rounded-xl p-4 border border-slate-700/20">
                  <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-yellow-500" />
                    Recommendations
                  </h3>
                  <ul className="space-y-2 text-sm text-slate-300">
                    {currentPrediction.predicted_wait_time > 30 && (
                      <li className="flex items-start gap-2">
                        <span className="text-orange-400 mt-1">•</span>
                        <span>High wait time detected. Consider adding more counters.</span>
                      </li>
                    )}
                    {inputData.system_utilization > 0.85 && (
                      <li className="flex items-start gap-2">
                        <span className="text-red-400 mt-1">•</span>
                        <span>System utilization is critical. Immediate action recommended.</span>
                      </li>
                    )}
                    {currentPrediction.predicted_wait_time <= 15 && (
                      <li className="flex items-start gap-2">
                        <span className="text-green-400 mt-1">•</span>
                        <span>Wait time is within acceptable range. System running smoothly.</span>
                      </li>
                    )}
                  </ul>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-slate-500">
                <div className="text-center">
                  <Clock className="w-16 h-16 mx-auto mb-4 opacity-30" />
                  <p>Enter parameters and click "Predict Wait Time"</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Counter Allocation Section */}
        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              <Activity className="w-6 h-6 text-green-400" />
              Smart Counter Allocation
            </h2>
            <button 
              onClick={handleAllocate}
              disabled={loading}
              className="bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white font-semibold py-3 px-6 rounded-xl transition-all duration-300 transform hover:scale-[1.02] disabled:opacity-50 shadow-lg shadow-green-500/20"
            >
              Calculate Optimal Allocation
            </button>
          </div>

          {allocations?.recommendations ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {Object.entries(allocations.recommendations).map(([dept, rec]) => (
                  <div key={dept} className="bg-slate-800/30 border border-slate-700/30 rounded-xl p-5 hover:border-slate-600/50 transition-all">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-bold text-white text-lg">{dept}</h3>
                      <div 
                        className="w-3 h-3 rounded-full animate-pulse"
                        style={{ backgroundColor: getAlertColor(rec.alert_level) }}
                      />
                    </div>

                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-400">Current:</span>
                        <span className="font-semibold text-slate-300">{rec.current_counters} counters</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-400">Recommended:</span>
                        <span className="font-bold text-cyan-400">{rec.recommended_counters} counters</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-400">Wait Time:</span>
                        <span className="font-semibold text-slate-300">{rec.predicted_wait_time} min</span>
                      </div>
                      <div className="pt-2 border-t border-slate-700/50">
                        <div className="text-xs text-slate-500 mb-1">Utilization</div>
                        <div className="w-full bg-slate-700/30 rounded-full h-2 overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-500"
                            style={{ width: `${Math.min(rec.utilization * 100, 100)}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 p-3 bg-slate-900/50 rounded-lg">
                      <p className="text-xs text-slate-400">{rec.reasoning}</p>
                    </div>
                  </div>
                ))}
              </div>

              {allocations.summary && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                  <div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border border-blue-500/20 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">Total Staff Recommended</div>
                    <div className="text-3xl font-bold text-white">
                      {allocations.summary.total_staff_recommended} / 15
                    </div>
                  </div>
                  <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/20 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">Avg Wait Time</div>
                    <div className="text-3xl font-bold text-white">
                      {allocations.summary.average_wait_time} min
                    </div>
                  </div>
                  <div className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/20 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">Critical Alerts</div>
                    <div className="text-3xl font-bold text-white">
                      {allocations.summary.alerts.critical.length}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center h-48 text-slate-500">
              <div className="text-center">
                <Activity className="w-16 h-16 mx-auto mb-4 opacity-30" />
                <p>Click "Calculate Optimal Allocation" to see recommendations</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Add Google Fonts */}
      <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700;900&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
    </div>
  );
};

export default Dashboard;