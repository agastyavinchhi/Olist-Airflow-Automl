import { Link, useLocation } from 'react-router-dom'

const LINKS = [
  { to: '/', label: 'Projects' },
  { to: '/simple-loop', label: 'In-class loop (baseline)' },
  { to: '/olist-loop', label: 'Olist Loop' },
]

export function AppNav() {
  const loc = useLocation()
  return (
    <nav className="flex flex-wrap items-center gap-3 border-b border-line bg-panel/80 px-4 py-2 text-sm">
      <span className="font-display text-base text-paper">AutoML Research Lab</span>
      <span className="text-line">|</span>
      {LINKS.map((l) => {
        const active =
          l.to === '/'
            ? loc.pathname === '/' || loc.pathname.startsWith('/projects')
            : loc.pathname === l.to
        return (
          <Link
            key={l.to}
            to={l.to}
            className={active ? 'text-teal' : 'text-mist hover:text-paper'}
          >
            {l.label}
          </Link>
        )
      })}
    </nav>
  )
}
