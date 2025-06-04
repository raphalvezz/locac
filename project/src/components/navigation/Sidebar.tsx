import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useUser } from '../../contexts/UserContext';
import { 
  Home, Compass, Users, BarChart, Settings, 
  Briefcase, TrendingUp, Award, User, Calculator 
} from 'lucide-react';

interface NavItem {
  label: string;
  icon: React.ReactNode;
  path: string;
}

const Sidebar: React.FC = () => {
  const location = useLocation();
  const { user } = useUser();
  
  const isActive = (path: string) => location.pathname === path;
  
  const navItems: NavItem[] = [
    { label: 'Home', icon: <Home size={20} />, path: '/' },
    { label: 'Explore', icon: <Compass size={20} />, path: '/explore' },
    { label: 'Network', icon: <Users size={20} />, path: '/network' },
    { label: 'Analytics', icon: <BarChart size={20} />, path: '/analytics' },
  ];

  const professionalItems: NavItem[] = [
    { label: 'Campaign Simulator', icon: <Calculator size={20} />, path: '/campaign-simulator' },
    { label: 'Campaigns', icon: <TrendingUp size={20} />, path: '/campaigns' },
    { label: 'Agency Hub', icon: <Briefcase size={20} />, path: '/agency-hub' },
    { label: 'Influencer Zone', icon: <Award size={20} />, path: '/influencer' },
  ];

  return (
    <div className="h-full py-4 flex flex-col bg-white dark:bg-gray-800 rounded-lg shadow">
      {/* User info */}
      <div className="px-4 py-2 mb-6">
        <Link to={`/profile/${user?.id}`} className="flex items-center space-x-3">
          <div className="h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center text-primary-700 overflow-hidden">
            {user?.profileImage ? (
              <img src={user.profileImage} alt={user.name} className="h-full w-full object-cover" />
            ) : (
              <User size={20} />
            )}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
              {user?.name || 'User Name'}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
              {user?.userType || 'Affiliate'} • {user?.connections || 0} connections
            </p>
          </div>
        </Link>
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 px-3 space-y-1">
        <div className="space-y-1">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`
                group flex items-center px-3 py-2 text-sm font-medium rounded-md
                ${isActive(item.path) 
                  ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300' 
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-white'}
              `}
            >
              <span className={`mr-3 ${isActive(item.path) ? 'text-primary-500' : 'text-gray-500 dark:text-gray-400 group-hover:text-gray-500'}`}>
                {item.icon}
              </span>
              {item.label}
            </Link>
          ))}
        </div>

        {/* Professional Section */}
        <div className="pt-6 mt-6 border-t border-gray-200 dark:border-gray-700">
          <h3 className="px-3 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
            Professional Tools
          </h3>
          <div className="mt-2 space-y-1">
            {professionalItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`
                  group flex items-center px-3 py-2 text-sm font-medium rounded-md
                  ${isActive(item.path) 
                    ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300' 
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-white'}
                `}
              >
                <span className={`mr-3 ${isActive(item.path) ? 'text-primary-500' : 'text-gray-500 dark:text-gray-400 group-hover:text-gray-500'}`}>
                  {item.icon}
                </span>
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </nav>

      {/* Footer */}
      <div className="px-3 mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
        <Link
          to="/settings"
          className="group flex items-center px-3 py-2 text-sm font-medium rounded-md text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-white"
        >
          <Settings className="mr-3 h-5 w-5 text-gray-500 dark:text-gray-400 group-hover:text-gray-500" />
          Settings
        </Link>
      </div>
    </div>
  );
};

export default Sidebar;