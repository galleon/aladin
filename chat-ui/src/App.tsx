import { useEffect } from 'react'
import { useStore } from './store'
import api, { getMe, getAgents, getDataDomains } from './api'
import Login from './components/Login'
import ChatView from './components/ChatView'

function App() {
  const { token, user, setAuth, setAgents, setDomains, selectAgent, logout } = useStore()

  useEffect(() => {
    if (token) {
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`
      // Verify token and load data
      getMe()
        .then(res => {
          setAuth(token, res.data)
          return Promise.all([getAgents(), getDataDomains()])
        })
        .then(([agentsRes, domainsRes]) => {
          setAgents(agentsRes.data)
          setDomains(domainsRes.data)

          // Check for agent_id in URL parameters
          const urlParams = new URLSearchParams(window.location.search)
          const agentIdParam = urlParams.get('agent_id')
          if (agentIdParam) {
            const agentId = parseInt(agentIdParam)
            const agent = agentsRes.data.find((a: any) => a.id === agentId)
            if (agent) {
              selectAgent(agent)
            }
          }
        })
        .catch(() => logout())
    }
  }, [])

  if (!token || !user) {
    return <Login />
  }

  return <ChatView />
}

export default App


