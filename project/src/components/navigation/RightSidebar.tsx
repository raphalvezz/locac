import React from 'react';
import { Link } from 'react-router-dom';
import { User, ChevronRight, Users, Bell } from 'lucide-react';

interface SuggestedUser {
  id: string;
  name: string;
  userType: string;
  profileImage?: string;
  mutualConnections: number;
}

interface TrendingTopic {
  id: string;
  name: string;
  postCount: number;
}

const RightSidebar: React.FC = () => {
  // Mock data for suggested connections
  const suggestedUsers: SuggestedUser[] = [
    { id: 'user1', name: 'Sarah Johnson', userType: 'Traffic Manager', profileImage: 'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg', mutualConnections: 12 },
    { id: 'user2', name: 'Miguel Lopez', userType: 'Agency Owner', profileImage: 'https://images.pexels.com/photos/2379005/pexels-photo-2379005.jpeg', mutualConnections: 8 },
    { id: 'user3', name: 'Priya Patel', userType: 'Affiliate', profileImage: 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg', mutualConnections: 5 },
  ];

  // Mock data for trending topics
  const trendingTopics: TrendingTopic[] = [
    { id: 'topic1', name: 'Performance Marketing', postCount: 342 },
    { id: 'topic2', name: 'AI in Advertising', postCount: 265 },
    { id: 'topic3', name: 'TikTok Strategy', postCount: 189 },
  ];

  return (
    <div className="space-y-6">
      {/* Suggested Connections */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">Suggested Connections</h2>
        </div>
        <div className="p-4 space-y-4">
          {suggestedUsers.map((user) => (
            <div key={user.id} className="flex items-center space-x-3">
              <div className="flex-shrink-0 h-10 w-10 rounded-full bg-primary-100 overflow-hidden">
                {user.profileImage ? (
                  <img src={user.profileImage} alt={user.name} className="h-full w-full object-cover" />
                ) : (
                  <div className="h-full w-full flex items-center justify-center">
                    <User className="h-5 w-5 text-primary-600" />
                  </div>
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 dark:text-white truncate">{user.name}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                  {user.userType} • {user.mutualConnections} mutual connections
                </p>
              </div>
              <button className="rounded-full p-1 bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 hover:bg-primary-100 dark:hover:bg-primary-800/40">
                <Users className="h-5 w-5" />
              </button>
            </div>
          ))}
          <Link 
            to="/network" 
            className="block mt-2 text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 font-medium"
          >
            View all suggestions
            <ChevronRight className="inline-block ml-1 h-4 w-4" />
          </Link>
        </div>
      </div>
      
      {/* Trending Topics */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">Trending in Marketing</h2>
        </div>
        <div className="p-4">
          <ul className="space-y-3">
            {trendingTopics.map((topic) => (
              <li key={topic.id}>
                <Link 
                  to={`/explore?topic=${encodeURIComponent(topic.name)}`}
                  className="block"
                >
                  <p className="text-sm font-medium text-gray-900 dark:text-white hover:text-primary-600 dark:hover:text-primary-400">
                    #{topic.name}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {topic.postCount} posts
                  </p>
                </Link>
              </li>
            ))}
          </ul>
          <Link 
            to="/explore" 
            className="block mt-4 text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 font-medium"
          >
            Explore more topics
            <ChevronRight className="inline-block ml-1 h-4 w-4" />
          </Link>
        </div>
      </div>
      
      {/* Upcoming Events */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">Upcoming Events</h2>
        </div>
        <div className="p-4">
          <div className="rounded-lg bg-primary-50 dark:bg-primary-900/20 p-3 border border-primary-100 dark:border-primary-800">
            <div className="flex items-center justify-between">
              <div className="text-xs font-medium text-primary-800 dark:text-primary-300 bg-primary-100 dark:bg-primary-800/50 px-2 py-0.5 rounded">
                WEBINAR
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Tomorrow, 2:00 PM
              </div>
            </div>
            <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
              Advanced Affiliate Strategies for 2025
            </h3>
            <div className="mt-2 flex items-center text-xs text-gray-500 dark:text-gray-400">
              <Bell className="h-3.5 w-3.5 mr-1" />
              <span>165 attending</span>
            </div>
          </div>
          <Link 
            to="/events" 
            className="block mt-3 text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 font-medium"
          >
            View all events
            <ChevronRight className="inline-block ml-1 h-4 w-4" />
          </Link>
        </div>
      </div>
    </div>
  );
};

export default RightSidebar;