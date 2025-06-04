import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useUser } from '../contexts/UserContext';
import { User, MapPin, Briefcase, Calendar, Link as LinkIcon, Users, Edit, Share2 } from 'lucide-react';
import PostCard from '../components/posts/PostCard';
import { Post } from '../types/post';

interface ProfileUser {
  id: string;
  name: string;
  userType: string;
  profileImage?: string;
  coverImage?: string;
  bio?: string;
  location?: string;
  company?: string;
  website?: string;
  joinDate?: string;
  connections: number;
  followers: number;
  following: number;
}

const ProfilePage: React.FC = () => {
  const { userId } = useParams<{ userId: string }>();
  const { user: currentUser } = useUser();
  const [profileUser, setProfileUser] = useState<ProfileUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [posts, setPosts] = useState<Post[]>([]);
  const [activeTab, setActiveTab] = useState('posts');
  
  const isOwnProfile = currentUser?.id === userId || userId === 'me';
  
  // Fetch profile data and posts
  useEffect(() => {
    const fetchProfileData = async () => {
      setLoading(true);
      try {
        // Simulate API call delay
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // In a real app, this would fetch user data from the backend
        // For demo purposes, we'll use mock data
        if (isOwnProfile && currentUser) {
          setProfileUser({
            id: currentUser.id,
            name: currentUser.name,
            userType: currentUser.userType,
            profileImage: currentUser.profileImage,
            coverImage: 'https://images.pexels.com/photos/3183153/pexels-photo-3183153.jpeg',
            bio: currentUser.bio || 'Performance marketing specialist with expertise in affiliate marketing and paid social campaigns.',
            location: 'San Francisco, CA',
            company: 'Growth Hackers Inc.',
            website: 'https://example.com',
            joinDate: 'January 2023',
            connections: 142,
            followers: 256,
            following: 184
          });
          
          // Mock posts for current user
          setPosts([
            {
              id: 'post101',
              author: {
                id: currentUser.id,
                name: currentUser.name,
                userType: currentUser.userType,
                profileImage: currentUser.profileImage
              },
              content: 'Just launched our newest campaign for a major e-commerce client. Seeing incredible CTRs already! #digitalmarketing #success',
              timestamp: new Date().getTime() - 86400000, // 1 day ago
              likes: 37,
              comments: 8,
              shares: 5,
              media: [
                {
                  type: 'image',
                  url: 'https://images.pexels.com/photos/7654053/pexels-photo-7654053.jpeg',
                  alt: 'Digital marketing dashboard with analytics'
                }
              ]
            },
            {
              id: 'post102',
              author: {
                id: currentUser.id,
                name: currentUser.name,
                userType: currentUser.userType,
                profileImage: currentUser.profileImage
              },
              content: 'Attended an amazing workshop on advanced targeting techniques today. So many new ideas to implement! Who else is exploring custom audience segmentation? Would love to connect.',
              timestamp: new Date().getTime() - 172800000, // 2 days ago
              likes: 24,
              comments: 11,
              shares: 2
            }
          ]);
        } else {
          // Mock data for other user profile
          const mockUser: ProfileUser = {
            id: 'user1',
            name: 'Sarah Johnson',
            userType: 'Traffic Manager',
            profileImage: 'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg',
            coverImage: 'https://images.pexels.com/photos/7689843/pexels-photo-7689843.jpeg',
            bio: 'Traffic Manager with 6+ years experience specializing in performance marketing and conversion rate optimization.',
            location: 'New York, NY',
            company: 'Digital Surge Agency',
            website: 'https://example.com/sarahjohnson',
            joinDate: 'March 2022',
            connections: 276,
            followers: 312,
            following: 148
          };
          setProfileUser(mockUser);
          
          // Mock posts for other user
          setPosts([
            {
              id: 'post201',
              author: {
                id: 'user1',
                name: 'Sarah Johnson',
                userType: 'Traffic Manager',
                profileImage: 'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg'
              },
              content: 'Just published a new case study on our latest campaign that achieved a 250% ROI for our client in the finance sector. Check it out! #performancemarketing #paidmedia',
              timestamp: new Date().getTime() - 3600000, // 1 hour ago
              likes: 24,
              comments: 5,
              shares: 3,
              media: [
                {
                  type: 'image',
                  url: 'https://images.pexels.com/photos/7947541/pexels-photo-7947541.jpeg',
                  alt: 'Digital marketing analytics dashboard'
                }
              ]
            },
            {
              id: 'post202',
              author: {
                id: 'user1',
                name: 'Sarah Johnson',
                userType: 'Traffic Manager',
                profileImage: 'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg'
              },
              content: 'Excited to be speaking at the Digital Marketing Summit next month! Will be covering advanced targeting strategies for e-commerce. Who else will be attending? #dmsummit #speakingevent',
              timestamp: new Date().getTime() - 259200000, // 3 days ago
              likes: 46,
              comments: 18,
              shares: 7
            }
          ]);
        }
      } catch (error) {
        console.error('Error fetching profile data:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchProfileData();
  }, [userId, currentUser, isOwnProfile]);

  if (loading) {
    return (
      <div className="flex justify-center items-center p-8">
        <div className="animate-pulse-slow">
          <div className="h-12 w-12 rounded-full bg-primary-200 dark:bg-primary-700"></div>
        </div>
      </div>
    );
  }

  if (!profileUser) {
    return (
      <div className="p-8 text-center">
        <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200">User not found</h2>
        <p className="mt-2 text-gray-600 dark:text-gray-400">The user you're looking for doesn't exist or has been removed.</p>
      </div>
    );
  }

  return (
    <div className="pb-16 lg:pb-0">
      {/* Cover photo */}
      <div className="relative h-48 sm:h-64 w-full rounded-lg overflow-hidden bg-gray-200 dark:bg-gray-700">
        {profileUser.coverImage ? (
          <img 
            src={profileUser.coverImage} 
            alt={`${profileUser.name}'s cover`} 
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-r from-primary-500 to-secondary-500 opacity-75"></div>
        )}
        
        {/* Profile actions */}
        <div className="absolute top-4 right-4 flex space-x-2">
          <button className="bg-white dark:bg-gray-800 p-2 rounded-full shadow-md text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-white">
            <Share2 size={18} />
          </button>
          {isOwnProfile && (
            <button className="bg-white dark:bg-gray-800 p-2 rounded-full shadow-md text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-white">
              <Edit size={18} />
            </button>
          )}
        </div>
      </div>
      
      {/* Profile header */}
      <div className="relative px-4 sm:px-6 -mt-16">
        <div className="flex flex-col sm:flex-row sm:items-end sm:space-x-5">
          {/* Profile image */}
          <div className="relative inline-block">
            <div className="h-24 w-24 sm:h-32 sm:w-32 rounded-full ring-4 ring-white dark:ring-gray-800 bg-white dark:bg-gray-800 overflow-hidden">
              {profileUser.profileImage ? (
                <img 
                  src={profileUser.profileImage} 
                  alt={profileUser.name} 
                  className="h-full w-full object-cover"
                />
              ) : (
                <div className="h-full w-full flex items-center justify-center bg-primary-100 dark:bg-primary-900">
                  <User className="h-12 w-12 text-primary-600 dark:text-primary-400" />
                </div>
              )}
            </div>
            {isOwnProfile && (
              <button className="absolute bottom-1 right-1 bg-primary-500 text-white p-1.5 rounded-full shadow-md hover:bg-primary-600">
                <Edit size={14} />
              </button>
            )}
          </div>
          
          {/* Profile info and actions */}
          <div className="mt-6 sm:mt-0 flex-1 flex flex-col sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">{profileUser.name}</h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">{profileUser.userType}</p>
            </div>
            
            <div className="mt-4 sm:mt-0">
              {isOwnProfile ? (
                <button className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500">
                  Edit Profile
                </button>
              ) : (
                <button className="px-4 py-2 rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500">
                  Connect
                </button>
              )}
            </div>
          </div>
        </div>
        
        {/* Bio and details */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            {/* Bio */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">About</h2>
              <p className="text-gray-700 dark:text-gray-300">{profileUser.bio}</p>
              
              {/* Details */}
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex items-center text-gray-600 dark:text-gray-400">
                  <MapPin className="h-5 w-5 mr-2" />
                  <span>{profileUser.location}</span>
                </div>
                <div className="flex items-center text-gray-600 dark:text-gray-400">
                  <Briefcase className="h-5 w-5 mr-2" />
                  <span>{profileUser.company}</span>
                </div>
                <div className="flex items-center text-gray-600 dark:text-gray-400">
                  <LinkIcon className="h-5 w-5 mr-2" />
                  <a href={profileUser.website} className="text-primary-600 dark:text-primary-400 hover:underline" target="_blank" rel="noopener noreferrer">
                    {profileUser.website?.replace(/https?:\/\//, '')}
                  </a>
                </div>
                <div className="flex items-center text-gray-600 dark:text-gray-400">
                  <Calendar className="h-5 w-5 mr-2" />
                  <span>Joined {profileUser.joinDate}</span>
                </div>
              </div>
              
              {/* Stats */}
              <div className="mt-6 flex space-x-6 border-t border-gray-200 dark:border-gray-700 pt-4">
                <div className="text-center">
                  <span className="block text-lg font-semibold text-gray-900 dark:text-white">{profileUser.connections}</span>
                  <span className="text-sm text-gray-500 dark:text-gray-400">Connections</span>
                </div>
                <div className="text-center">
                  <span className="block text-lg font-semibold text-gray-900 dark:text-white">{profileUser.followers}</span>
                  <span className="text-sm text-gray-500 dark:text-gray-400">Followers</span>
                </div>
                <div className="text-center">
                  <span className="block text-lg font-semibold text-gray-900 dark:text-white">{profileUser.following}</span>
                  <span className="text-sm text-gray-500 dark:text-gray-400">Following</span>
                </div>
              </div>
            </div>
            
            {/* Tabs */}
            <div className="mt-6">
              <div className="border-b border-gray-200 dark:border-gray-700">
                <nav className="flex space-x-8">
                  <button
                    onClick={() => setActiveTab('posts')}
                    className={`py-4 px-1 border-b-2 font-medium text-sm ${
                      activeTab === 'posts'
                        ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                    }`}
                  >
                    Posts
                  </button>
                  <button
                    onClick={() => setActiveTab('articles')}
                    className={`py-4 px-1 border-b-2 font-medium text-sm ${
                      activeTab === 'articles'
                        ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                    }`}
                  >
                    Articles
                  </button>
                  <button
                    onClick={() => setActiveTab('activity')}
                    className={`py-4 px-1 border-b-2 font-medium text-sm ${
                      activeTab === 'activity'
                        ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                    }`}
                  >
                    Activity
                  </button>
                </nav>
              </div>
              
              {/* Tab content */}
              <div className="mt-6">
                {activeTab === 'posts' && (
                  <div className="space-y-4">
                    {posts.length > 0 ? (
                      posts.map((post) => (
                        <PostCard key={post.id} post={post} />
                      ))
                    ) : (
                      <div className="text-center py-6">
                        <p className="text-gray-500 dark:text-gray-400">No posts yet</p>
                      </div>
                    )}
                  </div>
                )}
                
                {activeTab === 'articles' && (
                  <div className="text-center py-6">
                    <p className="text-gray-500 dark:text-gray-400">No articles published yet</p>
                  </div>
                )}
                
                {activeTab === 'activity' && (
                  <div className="text-center py-6">
                    <p className="text-gray-500 dark:text-gray-400">Recent activity will appear here</p>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* Sidebar */}
          <div className="space-y-6">
            {/* Connections */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Connections</h2>
                <a href="#" className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300">
                  See all
                </a>
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div className="flex flex-col items-center">
                  <div className="h-12 w-12 rounded-full bg-primary-100 overflow-hidden">
                    <img src="https://images.pexels.com/photos/2379005/pexels-photo-2379005.jpeg" alt="Connection" className="h-full w-full object-cover" />
                  </div>
                  <span className="mt-1 text-xs text-gray-700 dark:text-gray-300 truncate w-full text-center">Miguel L.</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="h-12 w-12 rounded-full bg-primary-100 overflow-hidden">
                    <img src="https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg" alt="Connection" className="h-full w-full object-cover" />
                  </div>
                  <span className="mt-1 text-xs text-gray-700 dark:text-gray-300 truncate w-full text-center">Priya P.</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="h-12 w-12 rounded-full bg-primary-100 overflow-hidden">
                    <img src="https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg" alt="Connection" className="h-full w-full object-cover" />
                  </div>
                  <span className="mt-1 text-xs text-gray-700 dark:text-gray-300 truncate w-full text-center">David W.</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="h-12 w-12 rounded-full bg-primary-100 overflow-hidden">
                    <img src="https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg" alt="Connection" className="h-full w-full object-cover" />
                  </div>
                  <span className="mt-1 text-xs text-gray-700 dark:text-gray-300 truncate w-full text-center">Emma T.</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="h-12 w-12 rounded-full bg-primary-100 overflow-hidden">
                    <img src="https://images.pexels.com/photos/614810/pexels-photo-614810.jpeg" alt="Connection" className="h-full w-full object-cover" />
                  </div>
                  <span className="mt-1 text-xs text-gray-700 dark:text-gray-300 truncate w-full text-center">Alex K.</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="h-12 w-12 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                    <span className="text-xs text-gray-500 dark:text-gray-400">+{profileUser.connections - 5}</span>
                  </div>
                  <span className="mt-1 text-xs text-gray-700 dark:text-gray-300 truncate w-full text-center">More</span>
                </div>
              </div>
            </div>
            
            {/* Skills */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Skills & Expertise</h2>
                {isOwnProfile && (
                  <button className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300">
                    Edit
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm">
                  Performance Marketing
                </span>
                <span className="px-3 py-1 bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm">
                  Paid Social
                </span>
                <span className="px-3 py-1 bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm">
                  Affiliate Marketing
                </span>
                <span className="px-3 py-1 bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm">
                  SEO
                </span>
                <span className="px-3 py-1 bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm">
                  Analytics
                </span>
                <span className="px-3 py-1 bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm">
                  Content Strategy
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;