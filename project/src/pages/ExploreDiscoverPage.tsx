import React, { useState, useEffect } from 'react';
import { Search, Compass, TrendingUp, Users, Filter, X } from 'lucide-react';
import { Post } from '../types/post';
import PostCard from '../components/posts/PostCard';

interface Category {
  id: string;
  name: string;
  icon: React.ReactNode;
}

interface Topic {
  id: string;
  name: string;
  postCount: number;
}

const ExploreDiscoverPage: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [trendingPosts, setTrendingPosts] = useState<Post[]>([]);
  const [trendingTopics, setTrendingTopics] = useState<Topic[]>([]);
  const [peopleToFollow, setPeopleToFollow] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    userType: '',
    sortBy: 'trending'
  });
  const [showFilters, setShowFilters] = useState(false);
  
  // Categories for exploration
  const categories: Category[] = [
    { id: 'performance', name: 'Performance Marketing', icon: <TrendingUp className="h-5 w-5" /> },
    { id: 'social', name: 'Social Media', icon: <AtSign className="h-5 w-5" /> },
    { id: 'affiliate', name: 'Affiliate Marketing', icon: <Compass className="h-5 w-5" /> },
    { id: 'agencies', name: 'Agencies', icon: <Users className="h-5 w-5" /> },
    { id: 'influencer', name: 'Influencer Marketing', icon: <Star className="h-5 w-5" /> },
    { id: 'analytics', name: 'Analytics & Data', icon: <BarChart className="h-5 w-5" /> },
  ];
  
  // Load explore data
  useEffect(() => {
    const fetchExploreData = async () => {
      setLoading(true);
      try {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // Mock trending posts data
        const mockTrendingPosts: Post[] = [
          {
            id: 'post1',
            author: {
              id: 'user1',
              name: 'Sarah Johnson',
              userType: 'Traffic Manager',
              profileImage: 'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg'
            },
            content: 'Just published a new case study on our latest campaign that achieved a 250% ROI for our client in the finance sector. Check it out! #performancemarketing #paidmedia',
            timestamp: Date.now() - 3600000, // 1 hour ago
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
            id: 'post2',
            author: {
              id: 'user2',
              name: 'Miguel Lopez',
              userType: 'Agency Owner',
              profileImage: 'https://images.pexels.com/photos/2379005/pexels-photo-2379005.jpeg'
            },
            content: 'Looking for experienced affiliate marketers to join our rapidly growing team. We\'re expanding our operations in the health and wellness niche. DM me if interested! #hiring #affiliatemarketing',
            timestamp: Date.now() - 7200000, // 2 hours ago
            likes: 18,
            comments: 12,
            shares: 7
          },
          {
            id: 'post3',
            author: {
              id: 'user3',
              name: 'Priya Patel',
              userType: 'Affiliate',
              profileImage: 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg'
            },
            content: 'Facebook\'s new ad policy changes are going to significantly impact how we target audiences. Here\'s what you need to know and how to adapt your strategies:',
            timestamp: Date.now() - 18000000, // 5 hours ago
            likes: 56,
            comments: 23,
            shares: 14,
            media: [
              {
                type: 'image',
                url: 'https://images.pexels.com/photos/5849592/pexels-photo-5849592.jpeg',
                alt: 'Social media marketing on mobile devices'
              }
            ]
          },
          {
            id: 'post4',
            author: {
              id: 'user4',
              name: 'David Wilson',
              userType: 'Influencer',
              profileImage: 'https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg'
            },
            content: 'Excited to announce my partnership with @TechBrand for their upcoming product launch! We\'ve been working on some amazing content that I can\'t wait to share with you all. #sponsored #influencermarketing',
            timestamp: Date.now() - 36000000, // 10 hours ago
            likes: 87,
            comments: 16,
            shares: 9,
            media: [
              {
                type: 'image',
                url: 'https://images.pexels.com/photos/3182812/pexels-photo-3182812.jpeg',
                alt: 'Person working with camera and laptop'
              }
            ]
          }
        ];
        
        // Mock trending topics
        const mockTrendingTopics: Topic[] = [
          { id: 'topic1', name: 'Performance Marketing', postCount: 342 },
          { id: 'topic2', name: 'AI in Advertising', postCount: 265 },
          { id: 'topic3', name: 'TikTok Strategy', postCount: 189 },
          { id: 'topic4', name: 'Affiliate Programs', postCount: 156 },
          { id: 'topic5', name: 'Data Privacy', postCount: 143 },
          { id: 'topic6', name: 'Conversion Optimization', postCount: 124 },
        ];
        
        // Mock people to follow
        const mockPeopleToFollow = [
          {
            id: 'user5',
            name: 'Emma Thompson',
            userType: 'SEO Specialist',
            profileImage: 'https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg',
            followers: 3245
          },
          {
            id: 'user6',
            name: 'Alex Kim',
            userType: 'Digital Marketing Director',
            profileImage: 'https://images.pexels.com/photos/614810/pexels-photo-614810.jpeg',
            followers: 8732
          },
          {
            id: 'user7',
            name: 'Sophia Rodriguez',
            userType: 'Influencer Manager',
            profileImage: 'https://images.pexels.com/photos/1036623/pexels-photo-1036623.jpeg',
            followers: 5621
          }
        ];
        
        setTrendingPosts(mockTrendingPosts);
        setTrendingTopics(mockTrendingTopics);
        setPeopleToFollow(mockPeopleToFollow);
      } catch (error) {
        console.error('Error fetching explore data:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchExploreData();
  }, []);
  
  // Filter posts based on selected category
  const filteredPosts = selectedCategory
    ? trendingPosts.filter(post => post.content.toLowerCase().includes(selectedCategory.toLowerCase()))
    : trendingPosts;
  
  // Handle search
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    // In a real app, this would trigger an API call with the search query
    console.log('Searching for:', searchQuery);
  };
  
  // Update filters
  const handleFilterChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFilters(prev => ({ ...prev, [name]: value }));
  };
  
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
      {/* Search and Filter Bar */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 mb-6">
        <form onSubmit={handleSearch}>
          <div className="flex items-center">
            <div className="relative flex-1">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                placeholder="Search posts, people, or topics..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-white"
              />
            </div>
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className="ml-2 p-2 text-gray-500 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md"
            >
              <Filter className="h-5 w-5" />
            </button>
          </div>
          
          {/* Filters */}
          {showFilters && (
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="userType" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  User Type
                </label>
                <select
                  id="userType"
                  name="userType"
                  value={filters.userType}
                  onChange={handleFilterChange}
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md dark:bg-gray-700 dark:text-white"
                >
                  <option value="">All Types</option>
                  <option value="Affiliate">Affiliates</option>
                  <option value="Traffic Manager">Traffic Managers</option>
                  <option value="Agency">Agencies</option>
                  <option value="Company">Companies</option>
                  <option value="Influencer">Influencers</option>
                </select>
              </div>
              <div>
                <label htmlFor="sortBy" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Sort By
                </label>
                <select
                  id="sortBy"
                  name="sortBy"
                  value={filters.sortBy}
                  onChange={handleFilterChange}
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md dark:bg-gray-700 dark:text-white"
                >
                  <option value="trending">Trending</option>
                  <option value="latest">Latest</option>
                  <option value="popular">Most Popular</option>
                </select>
              </div>
            </div>
          )}
        </form>
      </div>
      
      {/* Categories */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 mb-6 overflow-x-auto">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">Explore Categories</h3>
        <div className="flex space-x-3 pb-1">
          {categories.map((category) => (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(selectedCategory === category.id ? null : category.id)}
              className={`flex items-center px-4 py-2 rounded-full text-sm font-medium ${
                selectedCategory === category.id
                  ? 'bg-primary-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <span className="mr-1.5">{category.icon}</span>
              {category.name}
            </button>
          ))}
        </div>
      </div>
      
      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trending Posts - Left Side */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                {selectedCategory ? `Posts in ${categories.find(c => c.id === selectedCategory)?.name}` : 'Trending Posts'}
              </h2>
              {selectedCategory && (
                <button 
                  onClick={() => setSelectedCategory(null)}
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                >
                  <X className="h-5 w-5" />
                </button>
              )}
            </div>
            
            <div className="divide-y divide-gray-200 dark:divide-gray-700">
              {filteredPosts.length > 0 ? (
                filteredPosts.map((post) => (
                  <div key={post.id} className="p-4">
                    <PostCard post={post} />
                  </div>
                ))
              ) : (
                <div className="p-8 text-center">
                  <p className="text-gray-500 dark:text-gray-400">No posts found in this category.</p>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Right Sidebar */}
        <div className="space-y-6">
          {/* Trending Topics */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Trending Topics</h2>
            </div>
            <div className="p-4">
              <ul className="space-y-4">
                {trendingTopics.map((topic) => (
                  <li key={topic.id}>
                    <a 
                      href="#" 
                      className="flex items-center justify-between group"
                    >
                      <div>
                        <span className="text-sm font-medium text-gray-900 dark:text-white group-hover:text-primary-600 dark:group-hover:text-primary-400">
                          #{topic.name}
                        </span>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {topic.postCount} posts
                        </p>
                      </div>
                      <TrendingUp className="h-4 w-4 text-gray-400 group-hover:text-primary-500" />
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          
          {/* People to Follow */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">People to Follow</h2>
            </div>
            <div className="p-4">
              <ul className="space-y-4">
                {peopleToFollow.map((person) => (
                  <li key={person.id} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <div className="h-10 w-10 rounded-full bg-primary-100 overflow-hidden">
                        {person.profileImage ? (
                          <img src={person.profileImage} alt={person.name} className="h-full w-full object-cover" />
                        ) : (
                          <div className="h-full w-full flex items-center justify-center">
                            <User className="h-5 w-5 text-primary-600" />
                          </div>
                        )}
                      </div>
                      <div className="ml-3">
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          {person.name}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {person.userType} • {person.followers.toLocaleString()} followers
                        </p>
                      </div>
                    </div>
                    <button className="ml-2 px-3 py-1 text-xs font-medium rounded-full border border-primary-500 text-primary-600 dark:text-primary-400 hover:bg-primary-50 dark:hover:bg-primary-900/20">
                      Follow
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Import the AtSign, Star, and BarChart icons at the top level
import { AtSign, Star, BarChart } from 'lucide-react';

export default ExploreDiscoverPage;