import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from '../components/navigation/Navbar';
import Sidebar from '../components/navigation/Sidebar';
import RightSidebar from '../components/navigation/RightSidebar';
import MobileNavigation from '../components/navigation/MobileNavigation';

const AppLayout: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Navbar />
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Sidebar - Hidden on mobile */}
          <div className="hidden lg:block lg:w-64 flex-shrink-0">
            <Sidebar />
          </div>
          
          {/* Main content */}
          <main className="flex-1">
            <Outlet />
          </main>
          
          {/* Right sidebar - Hidden on mobile */}
          <div className="hidden lg:block lg:w-80 flex-shrink-0">
            <RightSidebar />
          </div>
        </div>
      </div>
      
      {/* Mobile navigation - Visible only on mobile */}
      <div className="block lg:hidden">
        <MobileNavigation />
      </div>
    </div>
  );
};

export default AppLayout;