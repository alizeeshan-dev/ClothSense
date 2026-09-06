import { Navigate, Route, Routes } from 'react-router-dom'
import FloatingNav from './components/FloatingNav'
import OverviewPage from './pages/OverviewPage'
import ClassifyPage from './pages/ClassifyPage'
import ExperimentsPage from './pages/ExperimentsPage'
import GalleryPage from './pages/GalleryPage'

export default function App() {
  return (
    <div className="app-shell">
      <main className="page-shell">
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/classify" element={<ClassifyPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/gallery" element={<GalleryPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <FloatingNav />
    </div>
  )
}
