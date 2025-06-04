import React from 'react';
import { Routes, Route } from 'react-router-dom';
import AppLayout from './layouts/AppLayout';
import HomePage from './pages/HomePage';
import ProfilePage from './pages/ProfilePage';
import MessagesPage from './pages/MessagesPage';
import NotificationsPage from './pages/NotificationsPage';
import ExploreDiscoverPage from './pages/ExploreDiscoverPage';
import CampaignSimulatorPage from './pages/CampaignSimulatorPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import { UserProvider } from './contexts/UserContext';
import ProtectedRoute from './components/auth/ProtectedRoute';

function App() {
  return (
    <UserProvider>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/" element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        }>
          <Route index element={<HomePage />} />
          <Route path="profile/:userId" element={<ProfilePage />} />
          <Route path="messages" element={<MessagesPage />} />
          <Route path="notifications" element={<NotificationsPage />} />
          <Route path="explore" element={<ExploreDiscoverPage />} />
          <Route path="campaign-simulator" element={<CampaignSimulatorPage />} />
        </Route>
      </Routes>
    </UserProvider>
  );
}

export default App;