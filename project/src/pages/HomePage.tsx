import React, { useState } from 'react';
import { useUser } from '../contexts/UserContext';
import PostCreationCard from '../components/posts/PostCreationCard';
import PostCard from '../components/posts/PostCard';
import { Post } from '../types/post';

const HomePage: React.FC = () => {
  const { user } = useUser();
  const [posts, setPosts] = useState<Post[]>([
    {
      id: 'post1',
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
      id: 'post2',
      author: {
        id: 'user2',
        name: 'Miguel Lopez',
        userType: 'Agency Owner',
        profileImage: 'https://images.pexels.com/photos/2379005/pexels-photo-2379005.jpeg'
      },
      content: 'Looking for experienced affiliate marketers to join our rapidly growing team. We\'re expanding our operations in the health and wellness niche. DM me if interested! #hiring #affiliatemarketing',
      timestamp: new Date().getTime() - 7200000, // 2 hours ago
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
      timestamp: new Date().getTime() - 18000000, // 5 hours ago
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
    }
  ]);

  const handleNewPost = (newPost: Post) => {
    setPosts([newPost, ...posts]);
  };

  return (
    <div className="pb-16 lg:pb-0">
      {/* Main content */}
      <div className="space-y-4">
        {/* Post creation card */}
        <PostCreationCard onPostCreated={handleNewPost} />
        
        {/* Feed posts */}
        <div className="space-y-4">
          {posts.map((post) => (
            <PostCard key={post.id} post={post} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default HomePage;