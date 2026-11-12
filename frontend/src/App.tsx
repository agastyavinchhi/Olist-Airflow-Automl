import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { OlistLoopPage } from './pages/OlistLoopPage'
import { ControlCenterPage } from './pages/ControlCenterPage'
import { ProjectDetailPage } from './pages/ProjectDetailPage'
import { ProjectsPage } from './pages/ProjectsPage'
import { SetupPage } from './pages/SetupPage'
import { SimpleLoopPage } from './pages/SimpleLoopPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ProjectsPage />} />
        <Route path="/projects/:projectId" element={<ProjectDetailPage />} />
        <Route path="/projects/:projectId/new" element={<SetupPage />} />
        <Route path="/session/:sessionId" element={<ControlCenterPage />} />
        <Route path="/simple-loop" element={<SimpleLoopPage />} />
        <Route path="/olist-loop" element={<OlistLoopPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
