import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Tenants from './pages/Tenants'
import TenantDetail from './pages/TenantDetail'
import CreateTenant from './pages/CreateTenant'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/tenants" element={<Tenants />} />
          <Route path="/tenants/new" element={<CreateTenant />} />
          <Route path="/tenants/:id" element={<TenantDetail />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App

