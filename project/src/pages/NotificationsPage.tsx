import React, { useState, useEffect } from 'react';
import { Bell, ThumbsUp, MessageSquare, Users, Award, Mail, Link as LinkIcon, Briefcase } from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';

interface Notification {
  id: string;
  type: 'like' | 'comment' | 'connection' | 'mention' | 'achievement' | 'message' | 'share' | 'opportunity';
  title: string;
  description?: string;
  timestamp: number;
  read: boolean;
  actionUser?: {
    id: string;
    name: string;
    profileImage?: string;
  };
  actionLink?: string;
}

const NotificationsPage: React.FC = () => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeFilter, setActiveFilter] = useState<string>('all');
  
  // Fetch notifications
  useEffect(() => {
    // Simulate API call to get notifications
    const fetchNotifications = async () => {
      setLoading(true);
      try {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // Mock notifications data
        const mockNotifications: Notification[] = [
          {
            id: 'notif1',
            type: 'like',
            title: 'Sarah Johnson liked your post',
            description: '"Just published a new case study on our latest campaign that achieved a 250% ROI for our client in the finance sector."',
            timestamp: Date.now() - 1800000, // 30 minutes ago
            read: false,
            actionUser: {
              id: 'user1',
              name: 'Sarah Johnson',
              profileImage: 'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg'
            },
            actionLink: '/post/123'
          },
          {
            id: 'notif2',
            type: 'comment',
            title: 'Miguel Lopez commented on your post',
            description: '"Great insights! Would love to hear more about the targeting strategy you used."',
            timestamp: Date.now() - 7200000, // 2 hours ago
            read: false,
            actionUser: {
              id: 'user2',
              name: 'Miguel Lopez',
              profileImage: 'https://images.pexels.com/photos/2379005/pexels-photo-2379005.jpeg'
            },
            actionLink: '/post/123#comments'
          },
          {
            id: 'notif3',
            type: 'connection',
            title: 'Priya Patel accepted your connection request',
            timestamp: Date.now() - 86400000, // 1 day ago
            read: true,
            actionUser: {
              id: 'user3',
              name: 'Priya Patel',
              profileImage: 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg'
            },
            actionLink: '/profile/user3'
          },
          {
            id: 'notif4',
            type: 'message',
            title: 'New message from David Wilson',
            description: '"Hey, I saw your recent campaign and would love to connect about a potential collaboration."',
            timestamp: Date.now() - 172800000, // 2 days ago
            read: true,
            actionUser: {
              id: 'user4',
              name: 'David Wilson',
              profileImage: 'https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg'
            },
            actionLink: '/messages/user4'
          },
          {
            id: 'notif5',
            type: 'mention',
            title: 'Emma Thompson mentioned you in a comment',
            description: '"@JaneSmith would be perfect for this kind of project!"',
            timestamp: Date.now() - 259200000, // 3 days ago
            read: true,
            actionUser: {
              id: 'user5',
              name: 'Emma Thompson',
              profileImage: 'https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg'
            },
            actionLink: '/post/456#comments'
          },
          {
            id: 'notif6',
            type: 'achievement',
            title: 'You reached 100+ profile views!',
            description: 'Your profile is getting noticed. Keep building your professional brand.',
            timestamp: Date.now() - 345600000, // 4 days ago
            read: true,
            actionLink: '/profile/analytics'
          },
          {
            id: 'notif7',
            type: 'share',
            title: 'Alex Kim shared your post',
            description: '"Check out this insightful analysis of the latest marketing trends by @JaneSmith"',
            timestamp: Date.now() - 432000000, // 5 days ago
            read: true,
            actionUser: {
              id: 'user6',
              name: 'Alex Kim',
              profileImage: 'https://images.pexels.com/photos/614810/pexels-photo-614810.jpeg'
            },
            actionLink: '/post/shared/789'
          },
          {
            id: 'notif8',
            type: 'opportunity',
            title: 'New job opportunity that matches your profile',
            description: 'Senior Traffic Manager at Digital Growth Agency',
            timestamp: Date.now() - 518400000, // 6 days ago
            read: true,
            actionLink: '/jobs/123'
          }
        ];
        
        setNotifications(mockNotifications);
      } catch (error) {
        console.error('Error fetching notifications:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchNotifications();
  }, []);
  
  // Mark notification as read
  const markAsRead = (id: string) => {
    setNotifications(prevNotifications => 
      prevNotifications.map(notification => 
        notification.id === id ? { ...notification, read: true } : notification
      )
    );
  };
  
  // Mark all notifications as read
  const markAllAsRead = () => {
    setNotifications(prevNotifications => 
      prevNotifications.map(notification => ({ ...notification, read: true }))
    );
  };
  
  // Format notification time
  const formatNotificationTime = (timestamp: number) => {
    const now = new Date();
    const notificationDate = new Date(timestamp);
    
    // If less than 24 hours ago, show relative time
    if (now.getTime() - timestamp < 86400000) {
      return formatDistanceToNow(notificationDate, { addSuffix: true });
    }
    
    // Otherwise, show the date
    return format(notificationDate, 'MMM d');
  };
  
  // Get notification icon based on type
  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'like':
        return <ThumbsUp className="h-5 w-5 text-primary-500" />;
      case 'comment':
        return <MessageSquare className="h-5 w-5 text-secondary-500" />;
      case 'connection':
        return <Users className="h-5 w-5 text-success-500" />;
      case 'mention':
        return <AtSign className="h-5 w-5 text-warning-500" />;
      case 'achievement':
        return <Award className="h-5 w-5 text-accent-500" />;
      case 'message':
        return <Mail className="h-5 w-5 text-primary-500" />;
      case 'share':
        return <LinkIcon className="h-5 w-5 text-secondary-500" />;
      case 'opportunity':
        return <Briefcase className="h-5 w-5 text-success-500" />;
      default:
        return <Bell className="h-5 w-5 text-gray-500" />;
    }
  };
  
  // Filter notifications
  const filteredNotifications = activeFilter === 'all' 
    ? notifications 
    : activeFilter === 'unread'
      ? notifications.filter(notification => !notification.read)
      : notifications.filter(notification => notification.type === activeFilter);
  
  // Count unread notifications
  const unreadCount = notifications.filter(notification => !notification.read).length;
  
  if (loading) {
    return (
      <div className="flex justify-center items-center p-8">
        <div className="animate-pulse-slow">
          <div className="h-12 w-12 rounded-full bg-primary-200 dark:bg-primary-700"></div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="pb-16 lg:pb-0">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Notifications</h2>
            {unreadCount > 0 && (
              <p className="text-sm text-gray-500 dark:text-gray-400">
                You have {unreadCount} unread notification{unreadCount !== 1 ? 's' : ''}
              </p>
            )}
          </div>
          <button
            onClick={markAllAsRead}
            className="text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300 font-medium"
          >
            Mark all as read
          </button>
        </div>
        
        {/* Filters */}
        <div className="border-b border-gray-200 dark:border-gray-700 p-2 overflow-x-auto scrollbar-hide">
          <div className="flex space-x-2">
            <button
              onClick={() => setActiveFilter('all')}
              className={`px-3 py-1.5 text-sm font-medium rounded-md ${
                activeFilter === 'all'
                  ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setActiveFilter('unread')}
              className={`px-3 py-1.5 text-sm font-medium rounded-md ${
                activeFilter === 'unread'
                  ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              Unread
            </button>
            <button
              onClick={() => setActiveFilter('connection')}
              className={`px-3 py-1.5 text-sm font-medium rounded-md ${
                activeFilter === 'connection'
                  ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              Connections
            </button>
            <button
              onClick={() => setActiveFilter('like')}
              className={`px-3 py-1.5 text-sm font-medium rounded-md ${
                activeFilter === 'like'
                  ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              Likes
            </button>
            <button
              onClick={() => setActiveFilter('comment')}
              className={`px-3 py-1.5 text-sm font-medium rounded-md ${
                activeFilter === 'comment'
                  ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              Comments
            </button>
            <button
              onClick={() => setActiveFilter('message')}
              className={`px-3 py-1.5 text-sm font-medium rounded-md ${
                activeFilter === 'message'
                  ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              Messages
            </button>
          </div>
        </div>
        
        {/* Notifications list */}
        {filteredNotifications.length > 0 ? (
          <ul className="divide-y divide-gray-200 dark:divide-gray-700">
            {filteredNotifications.map((notification) => (
              <li 
                key={notification.id} 
                className={`p-4 hover:bg-gray-50 dark:hover:bg-gray-700 ${
                  !notification.read ? 'bg-primary-50 dark:bg-primary-900/10' : ''
                }`}
                onClick={() => markAsRead(notification.id)}
              >
                <div className="flex items-start">
                  <div className="flex-shrink-0 mt-0.5">
                    <div className="h-10 w-10 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center overflow-hidden">
                      {notification.actionUser?.profileImage ? (
                        <img src={notification.actionUser.profileImage} alt={notification.actionUser.name} className="h-full w-full object-cover" />
                      ) : (
                        getNotificationIcon(notification.type)
                      )}
                    </div>
                  </div>
                  <div className="ml-3 flex-1">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      {notification.title}
                    </div>
                    {notification.description && (
                      <div className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                        {notification.description}
                      </div>
                    )}
                    <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                      {formatNotificationTime(notification.timestamp)}
                    </div>
                  </div>
                  {!notification.read && (
                    <div className="ml-2 flex-shrink-0">
                      <span className="inline-block h-2 w-2 rounded-full bg-primary-500"></span>
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <div className="py-12 text-center">
            <Bell className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500" />
            <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">No notifications</h3>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              {activeFilter === 'all' 
                ? 'You don\'t have any notifications yet.' 
                : `You don't have any ${activeFilter} notifications.`}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

// Import the AtSign icon at the top level
import { AtSign } from 'lucide-react';

export default NotificationsPage;