import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { tenantsApi, monitoringApi, billingApi, applicationsApi } from '../api/client'
import { ArrowLeft, Activity, DollarSign, AlertCircle, Package, Plus, Trash2 } from 'lucide-react'
import { XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { useState } from 'react'

export default function TenantDetail() {
  const { id } = useParams<{ id: string }>()
  const tenantId = parseInt(id || '0')

  const { data: tenant } = useQuery({
    queryKey: ['tenant', tenantId],
    queryFn: async () => {
      const response = await tenantsApi.getById(tenantId)
      return response.data
    },
  })

  const { data: status } = useQuery({
    queryKey: ['tenant-status', tenantId],
    queryFn: async () => {
      const response = await tenantsApi.getStatus(tenantId)
      return response.data
    },
    enabled: !!tenant,
    refetchInterval: 10000, // Refresh every 10 seconds
  })

  const { data: monitoring } = useQuery({
    queryKey: ['monitoring', tenantId],
    queryFn: async () => {
      const response = await monitoringApi.getTenantMonitoring(tenantId)
      return response.data
    },
    enabled: !!tenant,
    refetchInterval: 10000,
  })

  const { data: billingSummary } = useQuery({
    queryKey: ['billing-summary', tenantId],
    queryFn: async () => {
      const response = await billingApi.getSummary(tenantId)
      return response.data
    },
    enabled: !!tenant,
  })

  const { data: usageByModel } = useQuery({
    queryKey: ['billing-by-model', tenantId],
    queryFn: async () => {
      const response = await billingApi.getByModel(tenantId)
      return response.data
    },
    enabled: !!tenant,
  })

  const { data: applications } = useQuery({
    queryKey: ['applications'],
    queryFn: async () => {
      const response = await applicationsApi.getAll()
      return response.data
    },
  })

  const { data: deployedApplications } = useQuery({
    queryKey: ['deployed-applications', tenantId],
    queryFn: async () => {
      const response = await applicationsApi.getDeployments(tenantId)
      return response.data
    },
    enabled: !!tenant && tenant.status === 'active',
    refetchInterval: 10000,
  })

  const queryClient = useQueryClient()
  const [showDeployModal, setShowDeployModal] = useState(false)

  const deployMutation = useMutation({
    mutationFn: (data: { application_id: string; release_name?: string; values?: Record<string, any> }) =>
      applicationsApi.deploy(tenantId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['deployed-applications', tenantId] })
      setShowDeployModal(false)
    },
  })

  const uninstallMutation = useMutation({
    mutationFn: (deploymentId: number) => applicationsApi.uninstall(tenantId, deploymentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['deployed-applications', tenantId] })
    },
  })

  const handleDeploy = (applicationId: string) => {
    deployMutation.mutate({ application_id: applicationId })
  }

  const handleUninstall = (deploymentId: number, appName: string) => {
    if (window.confirm(`Are you sure you want to uninstall "${appName}"?`)) {
      uninstallMutation.mutate(deploymentId)
    }
  }

  if (!tenant) {
    return <div className="text-center py-12">Loading...</div>
  }

  return (
    <div>
      <Link
        to="/tenants"
        className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4"
      >
        <ArrowLeft className="w-4 h-4 mr-2" />
        Back to Tenants
      </Link>

      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">{tenant.name}</h1>
          <p className="text-gray-500 mt-1">Namespace: {tenant.namespace}</p>
        </div>
        <span
          className={`px-3 py-1 inline-flex text-sm leading-5 font-semibold rounded-full ${
            tenant.status === 'active'
              ? 'bg-green-100 text-green-800'
              : tenant.status === 'failed'
              ? 'bg-red-100 text-red-800'
              : 'bg-yellow-100 text-yellow-800'
          }`}
        >
          {tenant.status}
        </span>
      </div>

      {tenant.status === 'creating' && (
        <div className="mb-6 rounded-md bg-yellow-50 border border-yellow-200 p-4 text-sm text-yellow-800 flex items-start">
          <AlertCircle className="w-4 h-4 mr-2 mt-0.5" />
          <div>
            Provisioning in progress. We’ll update this page automatically every 10 seconds.
            Cluster creation with Cluster API can take a couple of minutes depending on your host machine.
          </div>
        </div>
      )}

      {tenant.status === 'failed' && (
        <div className="mb-6 rounded-md bg-red-50 border border-red-200 p-4 text-sm text-red-800 flex items-start">
          <AlertCircle className="w-4 h-4 mr-2 mt-0.5" />
          <div>
            Deployment failed. Check the provisioning status details below and backend logs for more information, then retry after addressing the issue.
          </div>
        </div>
      )}

      <div className="bg-white shadow rounded-lg p-6 mb-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
          <Activity className="w-5 h-5 mr-2" />
          Provisioning Status
        </h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <p className="text-sm font-medium text-gray-500">Cluster Phase</p>
            <p className="text-base font-semibold text-gray-900">
              {status?.cluster?.status?.phase || 'Unknown'}
            </p>
            {status?.cluster?.status?.conditions && status.cluster.status.conditions.length > 0 && (
              <ul className="mt-2 space-y-1 text-sm text-gray-600">
                {status.cluster.status.conditions.map((condition, idx) => (
                  <li key={idx} className="border border-gray-100 rounded px-3 py-2">
                    <div className="flex justify-between">
                      <span className="font-medium text-gray-700">{condition.type}</span>
                      <span
                        className={`text-xs uppercase font-semibold ${
                          condition.status === 'True'
                            ? 'text-green-600'
                            : condition.status === 'False'
                            ? 'text-red-600'
                            : 'text-yellow-600'
                        }`}
                      >
                        {condition.status}
                      </span>
                    </div>
                    {condition.reason && (
                      <div className="mt-1 text-xs text-gray-500">{condition.reason}</div>
                    )}
                    {condition.message && (
                      <div className="mt-1 text-xs text-gray-500">{condition.message}</div>
                    )}
                    {condition.lastTransitionTime && (
                      <div className="mt-1 text-xs text-gray-400">
                        Updated {new Date(condition.lastTransitionTime).toLocaleString()}
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="space-y-4">
            <div>
              <p className="text-sm font-medium text-gray-500">Namespace</p>
              <p className="text-base font-semibold text-gray-900">{status?.namespace || 'Unknown'}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-500">Helm Release</p>
              <p className="text-base font-semibold text-gray-900">
                {status?.helm?.info?.status || 'Not available yet'}
              </p>
              {status?.helm?.info?.notes && (
                <pre className="mt-2 whitespace-pre-wrap text-xs bg-gray-50 border border-gray-100 rounded p-2 text-gray-600">
                  {status.helm.info.notes}
                </pre>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2" />
            Deployment Status
          </h2>
          {status?.deployment ? (
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Replicas:</span>
                <span className="font-medium">
                  {status.deployment.readyReplicas || 0} / {status.deployment.replicas}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Available:</span>
                <span className="font-medium">{status.deployment.availableReplicas || 0}</span>
              </div>
            </div>
          ) : (
            <p className="text-gray-500">
              Deployment information will appear once the Helm release becomes ready.
            </p>
          )}
        </div>

        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <DollarSign className="w-5 h-5 mr-2" />
            Token Usage Summary
          </h2>
          {billingSummary ? (
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Input Tokens:</span>
                <span className="font-medium">{billingSummary.total_input_tokens || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Total Output Tokens:</span>
                <span className="font-medium">{billingSummary.total_output_tokens || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Total Tokens:</span>
                <span className="font-medium">{billingSummary.total_tokens || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Requests:</span>
                <span className="font-medium">{billingSummary.request_count || 0}</span>
              </div>
            </div>
          ) : (
            <p className="text-gray-500">No billing data available</p>
          )}
        </div>
      </div>

      {monitoring?.pods && monitoring.pods.length > 0 && (
        <div className="bg-white shadow rounded-lg p-6 mt-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Pods</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Containers</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {monitoring.pods.map((pod: any, idx: number) => (
                  <tr key={idx}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {pod.name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          pod.status === 'Running'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-yellow-100 text-yellow-800'
                        }`}
                      >
                        {pod.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      {pod.containers?.map((c: any) => c.name).join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {usageByModel && usageByModel.length > 0 && (
        <div className="bg-white shadow rounded-lg p-6 mt-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Token Usage by Model</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={usageByModel}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="total_input_tokens" fill="#8884d8" name="Input Tokens" />
              <Bar dataKey="total_output_tokens" fill="#82ca9d" name="Output Tokens" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {tenant.status === 'active' && (
        <div className="bg-white shadow rounded-lg p-6 mt-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-medium text-gray-900 flex items-center">
              <Package className="w-5 h-5 mr-2" />
              Applications
            </h2>
            {applications && applications.length > 0 && (
              <button
                onClick={() => setShowDeployModal(true)}
                className="inline-flex items-center px-3 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
              >
                <Plus className="w-4 h-4 mr-2" />
                Deploy Application
              </button>
            )}
          </div>

          {deployedApplications && deployedApplications.length > 0 ? (
            <div className="space-y-3">
              {deployedApplications.map((deployment) => {
                const app = applications?.find(a => a.id === deployment.application_id)
                return (
                  <div key={deployment.id} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex items-center">
                          <h3 className="text-base font-medium text-gray-900">
                            {app?.name || deployment.application_id}
                          </h3>
                          <span
                            className={`ml-3 px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                              deployment.status === 'active'
                                ? 'bg-green-100 text-green-800'
                                : deployment.status === 'failed'
                                ? 'bg-red-100 text-red-800'
                                : 'bg-yellow-100 text-yellow-800'
                            }`}
                          >
                            {deployment.status}
                          </span>
                        </div>
                        {app?.description && (
                          <p className="mt-1 text-sm text-gray-500">{app.description}</p>
                        )}
                        <div className="mt-2 text-sm text-gray-600">
                          <span className="font-medium">Release:</span> {deployment.release_name}
                        </div>
                        <div className="mt-1 text-sm text-gray-600">
                          <span className="font-medium">Version:</span> {app?.version || 'N/A'}
                        </div>
                      </div>
                      <button
                        onClick={() => handleUninstall(deployment.id, app?.name || deployment.application_id)}
                        className="ml-4 text-red-600 hover:text-red-800"
                        title="Uninstall"
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <Package className="w-12 h-12 mx-auto mb-3 text-gray-400" />
              <p>No applications deployed yet.</p>
              {applications && applications.length > 0 && (
                <p className="text-sm mt-2">Click "Deploy Application" to get started.</p>
              )}
            </div>
          )}

          {showDeployModal && (
            <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
              <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
                <div className="mt-3">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Deploy Application</h3>
                  {applications && applications.length > 0 ? (
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {applications.map((app) => {
                        const isDeployed = deployedApplications?.some(d => d.application_id === app.id)
                        return (
                          <button
                            key={app.id}
                            onClick={() => !isDeployed && handleDeploy(app.id)}
                            disabled={isDeployed || deployMutation.isPending}
                            className={`w-full text-left p-3 border rounded-lg transition-colors ${
                              isDeployed
                                ? 'bg-gray-50 border-gray-200 cursor-not-allowed opacity-60'
                                : deployMutation.isPending
                                ? 'bg-gray-50 border-gray-200 cursor-wait'
                                : 'border-gray-300 hover:border-blue-500 hover:bg-blue-50 cursor-pointer'
                            }`}
                          >
                            <div className="flex justify-between items-start">
                              <div>
                                <div className="font-medium text-gray-900">{app.name}</div>
                                {app.description && (
                                  <div className="text-sm text-gray-500 mt-1">{app.description}</div>
                                )}
                                <div className="text-xs text-gray-400 mt-1">Version: {app.version}</div>
                              </div>
                              {isDeployed && (
                                <span className="text-xs text-green-600 font-medium">Deployed</span>
                              )}
                            </div>
                          </button>
                        )
                      })}
                    </div>
                  ) : (
                    <p className="text-gray-500">No applications available.</p>
                  )}
                  <div className="mt-4 flex justify-end space-x-3">
                    <button
                      onClick={() => setShowDeployModal(false)}
                      className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

