import { FlaskConical, House, Images, ScanSearch } from 'lucide-react'
import { NavLink } from 'react-router-dom'

const navItems = [
  { to: '/', label: 'Overview', icon: House },
  { to: '/classify', label: 'Classify', icon: ScanSearch },
  { to: '/experiments', label: 'Experiments', icon: FlaskConical },
  { to: '/gallery', label: 'Gallery', icon: Images },
]

export default function FloatingNav() {
  return (
    <nav className="floating-nav" aria-label="Primary navigation">
      <div className="nav-brand" aria-label="ClothSense home">
        <span className="brand-mark">CS</span>
        <span className="brand-word">ClothSense</span>
      </div>
      <div className="nav-divider" aria-hidden="true" />
      <div className="nav-links">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            <Icon size={18} strokeWidth={1.8} aria-hidden="true" />
            <span>{label}</span>
          </NavLink>
        ))}
      </div>
    </nav>
  )
}
