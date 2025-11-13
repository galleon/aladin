import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { tenantsApi, tasksApi } from '../api/client'
import { Trash2, Eye, Loader2 } from 'lucide-react'

export default function Tenants() {
  const queryClient = useQueryClient()

  // Fetch active tasks to determine refetch interval
  const { data: activeTasks } = useQuery({
    queryKey: ['tasks', 'active'],
    queryFn: async () => {
      const response = await tasksApi.getAll({ limit: 100 })
      return response.data
    },
    refetchInterval: 2000, // Poll every 2 seconds
  })

  // Check if there are any active tenant tasks
  const hasActiveTenantTasks = activeTasks?.some(
    t => (t.task_type === 'create_tenant' || t.task_type === 'delete_tenant') &&
         (t.status === 'pending' || t.status === 'running')
  ) || false

  const { data: tenants, isLoading } = useQuery({
    queryKey: ['tenants'],
    queryFn: async () => {
      const response = await tenantsApi.getAll()
      return response.data
    },
    // Refresh more frequently when there are active tasks
    refetchInterval: hasActiveTenantTasks ? 2000 : false,
  })

  const deleteMutation = useMutation({
    mutationFn: (id: number) => tenantsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tenants'] })
    },
  })

  // Helper to get task progress for a tenant
  const getTenantTaskProgress = (tenantId: number, tenantStatus: string) => {
    // Check if tenant is in transitional state
    if (tenantStatus === 'creating' || tenantStatus === 'deleting') {
      // Find the active task for this tenant
      const task = activeTasks?.find(
        t => t.tenant_id === tenantId &&
             (t.status === 'pending' || t.status === 'running')
      )
      return task ? task.progress : 0
    }
    return null
  }

  const handleDelete = async (id: number, name: string) => {
    if (window.confirm(`Are you sure you want to delete tenant "${name}"? This will also delete the Kubernetes namespace and all resources.`)) {
      deleteMutation.mutate(id)
    }
  }

  if (isLoading) {
    return <div className="text-center py-12">Loading...</div>
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Tenants</h1>
        <Link
          to="/tenants/new"
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
        >
          Create New Tenant
        </Link>
      </div>

      {tenants && tenants.length > 0 ? (
        <div className="bg-white shadow overflow-hidden sm:rounded-md">
          <ul className="divide-y divide-gray-200">
            {tenants.map((tenant) => (
              <li key={tenant.id}>
                <div className="px-4 py-4 sm:px-6 flex items-center justify-between">
                  <div className="flex items-center">
                    <div>
                      <div className="flex items-center">
                        {getTenantTaskProgress(tenant.id, tenant.status) !== null && (
                          <div className="flex items-center mr-2">
                            <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                            <span className="ml-1 text-xs text-gray-600 font-medium">
                              {getTenantTaskProgress(tenant.id, tenant.status)}%
                            </span>
                          </div>
                        )}
                        <Link
                          to={`/tenants/${tenant.id}`}
                          className="text-sm font-medium text-blue-600 hover:text-blue-800"
                        >
                          {tenant.name}
                        </Link>
                        <span
                          className={`ml-3 px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
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
                      <div className="mt-2 space-y-2 sm:flex sm:justify-between sm:space-y-0 sm:gap-4">
                        <p className="text-sm text-gray-500">
                          Namespace: {tenant.namespace}
                        </p>
                        <p className="text-sm text-gray-500">
                          Created: {new Date(tenant.created_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Link
                      to={`/tenants/${tenant.id}`}
                      className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                    >
                      <Eye className="w-4 h-4 mr-2" />
                      View
                    </Link>
                    <button
                      onClick={() => handleDelete(tenant.id, tenant.name)}
                      disabled={deleteMutation.isPending}
                      className="inline-flex items-center px-3 py-2 border border-red-300 shadow-sm text-sm leading-4 font-medium rounded-md text-red-700 bg-white hover:bg-red-50 disabled:opacity-50"
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      Delete
                    </button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="bg-white shadow rounded-lg p-12 text-center">
          <p className="text-gray-500 mb-4">No tenants found.</p>
          <Link
            to="/tenants/new"
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
          >
            Create Your First Tenant
          </Link>
        </div>
      )}
    </div>
  )
}

