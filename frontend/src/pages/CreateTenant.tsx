import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { isAxiosError } from 'axios'
import { tenantsApi, CreateTenantRequest } from '../api/client'

export default function CreateTenant() {
    const navigate = useNavigate()
    const [formData, setFormData] = useState<CreateTenantRequest>({
        name: '',
        namespace: '',
        helmValues: {},
    })

  const createMutation = useMutation({
    mutationFn: (data: CreateTenantRequest) => tenantsApi.create(data),
    onSuccess: () => {
      // Task is queued, navigate immediately to show the tenant with throbber
      navigate('/tenants')
    },
  })

    const renderErrorMessage = () => {
        if (!createMutation.isError || !createMutation.error) {
            return 'Failed to create tenant'
        }

        const error = createMutation.error

        if (error instanceof Error && !isAxiosError(error)) {
            return error.message || 'Failed to create tenant'
        }

        if (isAxiosError(error)) {
            const detail = error.response?.data?.detail
            if (typeof detail === 'string' && detail.trim().length > 0) {
                return detail
            }

            if (typeof detail === 'object' && detail !== null) {
                const maybeMessage = (detail as { message?: string }).message
                if (maybeMessage) {
                    return maybeMessage
                }
            }

            if (error.message) {
                return error.message
            }
        }

        return 'Failed to create tenant'
    }

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        createMutation.mutate(formData)
    }

    return (
        <div className="max-w-2xl">
            <h1 className="text-3xl font-bold text-gray-900 mb-6">Create New Tenant</h1>

            <form onSubmit={handleSubmit} className="bg-white shadow rounded-lg p-6">
                <div className="mb-4">
                    <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
                        Tenant Name
                    </label>
                    <input
                        type="text"
                        id="name"
                        required
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                        placeholder="my-llm-platform"
                    />
                    <p className="mt-1 text-sm text-gray-500">
                        A unique name for your tenant
                    </p>
                </div>

                <div className="mb-4">
                    <label htmlFor="namespace" className="block text-sm font-medium text-gray-700 mb-2">
                        Kubernetes Namespace
                    </label>
                    <input
                        type="text"
                        id="namespace"
                        required
                        value={formData.namespace}
                        onChange={(e) => setFormData({ ...formData, namespace: e.target.value })}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                        placeholder="my-llm-platform"
                        pattern="[a-z0-9]([a-z0-9\-]*[a-z0-9])?"
                    />
                    <p className="mt-1 text-sm text-gray-500">
                        Must be a valid Kubernetes namespace name (lowercase alphanumeric and hyphens)
                    </p>
                </div>

                <div className="mb-6">
                    <label htmlFor="helmValues" className="block text-sm font-medium text-gray-700 mb-2">
                        Helm Values (JSON, optional)
                    </label>
                    <textarea
                        id="helmValues"
                        rows={8}
                        value={JSON.stringify(formData.helmValues || {}, null, 2)}
                        onChange={(e) => {
                            try {
                                const parsed = JSON.parse(e.target.value)
                                setFormData({ ...formData, helmValues: parsed })
                            } catch {
                                // Invalid JSON, ignore
                            }
                        }}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm font-mono"
                        placeholder='{\n  "replicas": 1,\n  "resources": {\n    "requests": {\n      "memory": "2Gi",\n      "cpu": "1000m"\n    }\n  }\n}'
                    />
                    <p className="mt-1 text-sm text-gray-500">
                        Optional Helm values to customize the LLM deployment
                    </p>
                </div>

                {createMutation.isSuccess && (
                    <div className="mb-4 bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded">
                        <p className="font-semibold">Tenant creation queued successfully!</p>
                        <p className="mt-1 text-sm">
                            The tenant is being created in the background. You'll see progress next to the tenant in the list.
                        </p>
                    </div>
                )}

                {createMutation.isError && (
                    <div className="mb-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                        <p className="font-semibold">Tenant creation failed.</p>
                        <p className="mt-1 text-sm">{renderErrorMessage()}</p>
                        <p className="mt-2 text-xs text-red-600">
                            Ensure the management cluster is running and the backend can access a valid kubeconfig.
                        </p>
                    </div>
                )}

                <div className="flex justify-end space-x-3">
                    <button
                        type="button"
                        onClick={() => navigate('/tenants')}
                        className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        disabled={createMutation.isPending}
                        className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50"
                    >
                        {createMutation.isPending ? 'Creating...' : 'Create Tenant'}
                    </button>
                </div>
            </form>
        </div>
    )
}

