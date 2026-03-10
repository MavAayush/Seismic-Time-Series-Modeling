'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Activity, Database, TrendingUp, AlertTriangle, Menu, X } from 'lucide-react'

interface DataRow {
  Datetime: string
  Latitude: number
  Longitude: number
  Depth: number
  Magnitude: number
}

interface ModelMetrics {
  R2: number
  MAE: number
  RMSE: number
  MAPE: number
  sMAPE: number
}

interface TrainingResults {
  status: string
  metrics: Record<string, ModelMetrics>
  timestamp: string
  data_points: number
}

interface AnomalyData {
  index: number
  datetime: string
  actual: number
  predicted: number
  residual: number
  z_score: number
}

interface AnomalyResults {
  anomaly_count: number
  best_model: string
  threshold: number
  anomalies: AnomalyData[]
  total_points: number
}

export default function App() {
  const [dataPreview, setDataPreview] = useState<DataRow[]>([])
  const [trainingResults, setTrainingResults] = useState<TrainingResults | null>(null)
  const [anomalyResults, setAnomalyResults] = useState<AnomalyResults | null>(null)
  const [loading, setLoading] = useState({ train: false, anomaly: false })
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // Load data preview on component mount
  useEffect(() => {
    fetchDataPreview()
  }, [])

  const fetchDataPreview = async () => {
    try {
      const response = await fetch('/api/data/head')
      const data = await response.json()
      setDataPreview(data.data)
    } catch (error) {
      console.error('Error fetching data preview:', error)
    }
  }

  const trainModels = async () => {
    setLoading({ ...loading, train: true })
    try {
      const response = await fetch('/api/train', { method: 'POST' })
      const results = await response.json()
      setTrainingResults(results)
    } catch (error) {
      console.error('Error training models:', error)
    } finally {
      setLoading({ ...loading, train: false })
    }
  }

  const detectAnomalies = async () => {
    setLoading({ ...loading, anomaly: true })
    try {
      const response = await fetch('/api/anomalies')
      const results = await response.json()
      setAnomalyResults(results)
    } catch (error) {
      console.error('Error detecting anomalies:', error)
    } finally {
      setLoading({ ...loading, anomaly: false })
    }
  }

  const scrollToSection = (sectionId: string) => {
    document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' })
    setMobileMenuOpen(false)
  }

  const formatMetricsForChart = (metrics: Record<string, ModelMetrics>) => {
    const metricNames = ['R2', 'MAE', 'RMSE', 'MAPE', 'sMAPE']
    return metricNames.map(metric => {
      const data: any = { metric }
      Object.keys(metrics).forEach(model => {
        data[model] = metrics[model][metric as keyof ModelMetrics]
      })
      return data
    })
  }

  const NavLink = ({ href, children }: { href: string; children: React.ReactNode }) => (
    <button
      onClick={() => scrollToSection(href)}
      className="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors"
    >
      {children}
    </button>
  )

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Navigation */}
      <nav className="bg-white shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Activity className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-xl font-bold text-gray-900">Seismic Modeling</h1>
            </div>
            
            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-4">
              <NavLink href="preview">Dataset</NavLink>
              <NavLink href="visualizations">Visualizations</NavLink>
              <NavLink href="training">Training</NavLink>
              <NavLink href="anomalies">Anomalies</NavLink>
            </div>

            {/* Mobile menu button */}
            <div className="md:hidden flex items-center">
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="text-gray-600 hover:text-blue-600"
              >
                {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </button>
            </div>
          </div>

          {/* Mobile Navigation */}
          {mobileMenuOpen && (
            <div className="md:hidden">
              <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-white border-t">
                <NavLink href="preview">Dataset</NavLink>
                <NavLink href="visualizations">Visualizations</NavLink>
                <NavLink href="training">Training</NavLink>
                <NavLink href="anomalies">Anomalies</NavLink>
              </div>
            </div>
          )}
        </div>
      </nav>

      {/* Hero Section */}
      <section className="section-padding text-center">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Seismic Time Series Modeling
          </h1>
          <p className="text-xl md:text-2xl text-gray-600 mb-8">
            Towards Earthquake Prediction
          </p>
          <p className="text-lg text-gray-700 max-w-3xl mx-auto">
            Advanced machine learning models including ARIMA, LSTM, and Hybrid approaches 
            for analyzing seismic data and detecting anomalies that could indicate potential earthquakes.
          </p>
        </div>
      </section>

      {/* Dataset Preview Section */}
      <section id="preview" className="section-padding bg-white">
        <div className="max-w-7xl mx-auto">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-6 w-6 text-blue-600" />
                Dataset Preview
              </CardTitle>
              <CardDescription>
                First 10 rows of the seismic dataset showing earthquake measurements
              </CardDescription>
            </CardHeader>
            <CardContent>
              {dataPreview.length > 0 ? (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Datetime</TableHead>
                        <TableHead>Latitude</TableHead>
                        <TableHead>Longitude</TableHead>
                        <TableHead>Depth (km)</TableHead>
                        <TableHead>Magnitude</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {dataPreview.map((row, index) => (
                        <TableRow key={index}>
                          <TableCell className="font-mono text-sm">{row.Datetime}</TableCell>
                          <TableCell>{row.Latitude.toFixed(4)}</TableCell>
                          <TableCell>{row.Longitude.toFixed(4)}</TableCell>
                          <TableCell>{row.Depth.toFixed(2)}</TableCell>
                          <TableCell className="font-semibold">{row.Magnitude.toFixed(2)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">Loading dataset...</div>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Visualizations Section */}
      <section id="visualizations" className="section-padding">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Data Visualizations</h2>
            <p className="text-lg text-gray-600">
              Comprehensive analysis of seismic patterns and distributions
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {[
              { type: 'magnitude_hist', title: 'Magnitude Distribution', desc: 'Frequency distribution of earthquake magnitudes' },
              { type: 'depth_hist', title: 'Depth Distribution', desc: 'Distribution of earthquake depths' },
              { type: 'corr_heatmap', title: 'Correlation Matrix', desc: 'Relationships between seismic features' },
              { type: 'scatter_locations', title: 'Geographic Distribution', desc: 'Earthquake locations with magnitude and depth' },
              { type: 'magnitude_time', title: 'Magnitude Over Time', desc: 'Temporal patterns in earthquake magnitudes' },
              { type: 'monthly_boxplot', title: 'Monthly Patterns', desc: 'Seasonal variations in earthquake activity' }
            ].map((plot) => (
              <Card key={plot.type} className="overflow-hidden">
                <CardHeader>
                  <CardTitle className="text-lg">{plot.title}</CardTitle>
                  <CardDescription>{plot.desc}</CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <img
                    src={`/api/visuals/${plot.type}`}
                    alt={plot.title}
                    className="w-full h-auto"
                    loading="lazy"
                  />
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Model Training Section */}
      <section id="training" className="section-padding bg-white">
        <div className="max-w-7xl mx-auto">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-6 w-6 text-green-600" />
                Model Training & Performance
              </CardTitle>
              <CardDescription>
                Train ARIMA, LSTM, and Hybrid models for earthquake prediction
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="mb-6">
                <Button 
                  onClick={trainModels} 
                  disabled={loading.train}
                  className="w-full sm:w-auto"
                >
                  {loading.train ? 'Training Models...' : 'Train All Models'}
                </Button>
              </div>

              {trainingResults && (
                <div className="space-y-6">
                  {/* Metrics Table */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4">Model Performance Metrics</h3>
                    <div className="overflow-x-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Model</TableHead>
                            <TableHead>R²</TableHead>
                            <TableHead>MAE</TableHead>
                            <TableHead>RMSE</TableHead>
                            <TableHead>MAPE (%)</TableHead>
                            <TableHead>sMAPE (%)</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {Object.entries(trainingResults.metrics).map(([model, metrics]) => (
                            <TableRow key={model}>
                              <TableCell className="font-semibold">{model}</TableCell>
                              <TableCell>{metrics.R2.toFixed(4)}</TableCell>
                              <TableCell>{metrics.MAE.toFixed(4)}</TableCell>
                              <TableCell>{metrics.RMSE.toFixed(4)}</TableCell>
                              <TableCell>{metrics.MAPE.toFixed(2)}</TableCell>
                              <TableCell>{metrics.sMAPE.toFixed(2)}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>

                  {/* Metrics Comparison Chart */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4">Performance Comparison</h3>
                    <div className="h-96">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={formatMetricsForChart(trainingResults.metrics)}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="metric" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="ARIMA" fill="#8884d8" />
                          <Bar dataKey="LSTM" fill="#82ca9d" />
                          <Bar dataKey="Hybrid" fill="#ffc658" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Training Info */}
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <p className="text-sm text-blue-800">
                      <strong>Training completed:</strong> {trainingResults.timestamp} | 
                      <strong> Data points:</strong> {trainingResults.data_points}
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Anomaly Detection Section */}
      <section id="anomalies" className="section-padding">
        <div className="max-w-7xl mx-auto">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-6 w-6 text-red-600" />
                Anomaly Detection
              </CardTitle>
              <CardDescription>
                Detect unusual seismic patterns that could indicate potential earthquakes
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="mb-6">
                <Button 
                  onClick={detectAnomalies} 
                  disabled={loading.anomaly || !trainingResults}
                  className="w-full sm:w-auto"
                >
                  {loading.anomaly ? 'Detecting Anomalies...' : 'Detect Anomalies'}
                </Button>
                {!trainingResults && (
                  <p className="text-sm text-gray-500 mt-2">
                    Please train models first to enable anomaly detection
                  </p>
                )}
              </div>

              {anomalyResults && (
                <div className="space-y-6">
                  {/* Summary */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-red-50 p-4 rounded-lg text-center">
                      <div className="text-2xl font-bold text-red-600">
                        {anomalyResults.anomaly_count}
                      </div>
                      <div className="text-sm text-red-800">Anomalies Detected</div>
                    </div>
                    <div className="bg-blue-50 p-4 rounded-lg text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {anomalyResults.best_model}
                      </div>
                      <div className="text-sm text-blue-800">Best Model Used</div>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg text-center">
                      <div className="text-2xl font-bold text-gray-600">
                        {anomalyResults.threshold}σ
                      </div>
                      <div className="text-sm text-gray-800">Detection Threshold</div>
                    </div>
                  </div>

                  {/* Anomaly Details Table */}
                  {anomalyResults.anomalies.length > 0 && (
                    <div>
                      <h3 className="text-lg font-semibold mb-4">Detected Anomalies</h3>
                      <div className="overflow-x-auto">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>DateTime</TableHead>
                              <TableHead>Actual</TableHead>
                              <TableHead>Predicted</TableHead>
                              <TableHead>Residual</TableHead>
                              <TableHead>Z-Score</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {anomalyResults.anomalies.map((anomaly, index) => (
                              <TableRow key={index}>
                                <TableCell className="font-mono text-sm">{anomaly.datetime}</TableCell>
                                <TableCell>{anomaly.actual.toFixed(3)}</TableCell>
                                <TableCell>{anomaly.predicted.toFixed(3)}</TableCell>
                                <TableCell className="text-red-600 font-semibold">
                                  {anomaly.residual.toFixed(3)}
                                </TableCell>
                                <TableCell className="text-red-600 font-semibold">
                                  {anomaly.z_score.toFixed(2)}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    </div>
                  )}

                  {/* Anomaly Plot */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4">Anomaly Visualization</h3>
                    <div className="bg-white rounded-lg overflow-hidden shadow-sm border">
                      <img
                        src="/api/anomalies/plot"
                        alt="Anomaly Detection Plot"
                        className="w-full h-auto"
                        loading="lazy"
                      />
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white section-padding">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center mb-4">
            <Activity className="h-8 w-8 text-blue-400 mr-3" />
            <h3 className="text-xl font-bold">Seismic Time Series Modeling</h3>
          </div>
          <p className="text-gray-400">
            Advanced earthquake prediction using machine learning and time series analysis
          </p>
        </div>
      </footer>
    </div>
  )
}