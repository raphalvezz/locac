import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, Compass, MessageSquare, Bell, User } from 'lucide-react';

interface NavItem {
  icon: React.ReactNode;
  label: string;
  path: string;
}

const MobileNavigation: React.FC = () => {
  const location = useLocation();
  const { pathname } = location;
  
  const isActive = (path: string) => pathname === path;

  const navItems: NavItem[] = [
    { icon: <Home size={24} />, label: 'Home', path: '/' },
    { icon: <Compass size={24} />, label: 'Explore', path: '/explore' },
    { icon: <MessageSquare size={24} />, label: 'Messages', path: '/messages' },
    { icon: <Bell size={24} />, label: 'Notifications', path: '/notifications' },
    { icon: <User size={24} />, label: 'Profile', path: '/profile/me' },
  ];

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 shadow-lg z-20">
      <div className="grid grid-cols-5">
        {navItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className="flex flex-col items-center justify-center h-16 text-xs"
          >
            <div 
              className={`p-1 rounded-full ${
                isActive(item.path) 
                  ? 'text-primary-600 dark:text-primary-400' 
                  : 'text-gray-500 dark:text-gray-400'
              }`}
            >
              {item.icon}
            </div>
            <span 
              className={`mt-1 ${
                isActive(item.path) 
                  ? 'text-primary-600 dark:text-primary-400 font-medium' 
                  : 'text-gray-500 dark:text-gray-400'
              }`}
            >
              {item.label}
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default MobileNavigation;