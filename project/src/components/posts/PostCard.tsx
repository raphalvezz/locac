import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { format, formatDistanceToNow } from 'date-fns';
import { ThumbsUp, MessageSquare, Share, MoreHorizontal, User } from 'lucide-react';
import { Post } from '../../types/post';

interface PostCardProps {
  post: Post;
}

const PostCard: React.FC<PostCardProps> = ({ post }) => {
  const [liked, setLiked] = useState(false);
  const [likes, setLikes] = useState(post.likes);
  const [showComments, setShowComments] = useState(false);
  
  // Format post time
  const formatPostTime = (timestamp: number) => {
    const postDate = new Date(timestamp);
    const now = new Date();
    
    // If less than 24 hours ago, show relative time
    if (now.getTime() - timestamp < 86400000) {
      return formatDistanceToNow(postDate, { addSuffix: true });
    }
    
    // Otherwise, show the date
    return format(postDate, 'MMM d');
  };
  
  // Toggle like
  const handleLike = () => {
    if (liked) {
      setLikes(likes - 1);
    } else {
      setLikes(likes + 1);
    }
    setLiked(!liked);
  };
  
  // Toggle comments
  const handleToggleComments = () => {
    setShowComments(!showComments);
  };
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
      {/* Post header */}
      <div className="p-4 flex items-start">
        <div className="flex-shrink-0">
          <Link to={`/profile/${post.author.id}`} className="block">
            <div className="h-10 w-10 rounded-full bg-primary-100 overflow-hidden">
              {post.author.profileImage ? (
                <img src={post.author.profileImage} alt={post.author.name} className="h-full w-full object-cover" />
              ) : (
                <div className="h-full w-full flex items-center justify-center">
                  <User className="h-5 w-5 text-primary-600" />
                </div>
              )}
            </div>
          </Link>
        </div>
        <div className="ml-3 flex-1">
          <div className="flex items-center justify-between">
            <div>
              <Link to={`/profile/${post.author.id}`} className="text-sm font-medium text-gray-900 dark:text-white hover:text-primary-600 dark:hover:text-primary-400">
                {post.author.name}
              </Link>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {post.author.userType} • {formatPostTime(post.timestamp)}
              </p>
            </div>
            <button className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300">
              <MoreHorizontal className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
      
      {/* Post content */}
      <div className="px-4 pb-2">
        <p className="text-gray-800 dark:text-gray-200 whitespace-pre-line">
          {post.content}
        </p>
      </div>
      
      {/* Post media */}
      {post.media && post.media.length > 0 && (
        <div className="mt-2">
          {post.media.map((media, index) => (
            <div key={index} className="relative">
              {media.type === 'image' && (
                <img
                  src={media.url}
                  alt={media.alt || 'Post image'}
                  className="w-full h-auto object-cover max-h-96"
                />
              )}
              {media.type === 'video' && (
                <video
                  src={media.url}
                  controls
                  className="w-full h-auto max-h-96"
                  poster={media.poster}
                />
              )}
            </div>
          ))}
        </div>
      )}
      
      {/* Post stats */}
      <div className="px-4 py-2 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 border-b border-gray-100 dark:border-gray-700">
        {likes > 0 && (
          <div className="flex items-center">
            <span className="flex items-center justify-center h-4 w-4 rounded-full bg-primary-100 dark:bg-primary-900/30">
              <ThumbsUp className="h-2.5 w-2.5 text-primary-600 dark:text-primary-400" />
            </span>
            <span className="ml-1">{likes}</span>
          </div>
        )}
        <div className="flex space-x-4">
          {post.comments > 0 && (
            <button 
              onClick={handleToggleComments}
              className="hover:text-gray-700 dark:hover:text-gray-300"
            >
              {post.comments} comments
            </button>
          )}
          {post.shares > 0 && (
            <span>{post.shares} shares</span>
          )}
        </div>
      </div>
      
      {/* Post actions */}
      <div className="flex items-center justify-around px-4 py-2">
        <button 
          onClick={handleLike}
          className={`flex items-center justify-center space-x-1 py-1.5 px-3 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 ${
            liked ? 'text-primary-600 dark:text-primary-400' : 'text-gray-500 dark:text-gray-400'
          }`}
        >
          <ThumbsUp className="h-5 w-5" />
          <span className="text-sm font-medium">Like</span>
        </button>
        <button 
          onClick={handleToggleComments}
          className="flex items-center justify-center space-x-1 py-1.5 px-3 rounded-md text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          <MessageSquare className="h-5 w-5" />
          <span className="text-sm font-medium">Comment</span>
        </button>
        <button className="flex items-center justify-center space-x-1 py-1.5 px-3 rounded-md text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700">
          <Share className="h-5 w-5" />
          <span className="text-sm font-medium">Share</span>
        </button>
      </div>
      
      {/* Comments section - shown when toggled */}
      {showComments && (
        <div className="px-4 pt-2 pb-4 border-t border-gray-100 dark:border-gray-700">
          {/* Comment input */}
          <div className="flex items-center mb-4">
            <div className="h-8 w-8 rounded-full bg-primary-100 overflow-hidden flex-shrink-0">
              <User className="h-full w-full p-1.5 text-primary-600" />
            </div>
            <div className="ml-2 flex-1 relative">
              <input
                type="text"
                placeholder="Write a comment..."
                className="w-full px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded-full text-sm bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>
          
          {/* Comments list - placeholder for now */}
          <div className="space-y-3">
            <div className="flex">
              <div className="h-8 w-8 rounded-full bg-primary-100 overflow-hidden flex-shrink-0">
                <img src="https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg" className="h-full w-full object-cover" alt="Commenter" />
              </div>
              <div className="ml-2">
                <div className="bg-gray-100 dark:bg-gray-700 rounded-2xl px-3 py-2">
                  <p className="text-xs font-medium text-gray-900 dark:text-white">Emma Thompson</p>
                  <p className="text-sm text-gray-800 dark:text-gray-200">Great insights! Would love to hear more about the targeting strategy you used.</p>
                </div>
                <div className="flex items-center mt-1 space-x-2 text-xs text-gray-500 dark:text-gray-400">
                  <button className="font-medium hover:text-gray-700 dark:hover:text-gray-300">Like</button>
                  <button className="font-medium hover:text-gray-700 dark:hover:text-gray-300">Reply</button>
                  <span>3h</span>
                </div>
              </div>
            </div>
            
            <div className="flex">
              <div className="h-8 w-8 rounded-full bg-primary-100 overflow-hidden flex-shrink-0">
                <img src="https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg" className="h-full w-full object-cover" alt="Commenter" />
              </div>
              <div className="ml-2">
                <div className="bg-gray-100 dark:bg-gray-700 rounded-2xl px-3 py-2">
                  <p className="text-xs font-medium text-gray-900 dark:text-white">David Wilson</p>
                  <p className="text-sm text-gray-800 dark:text-gray-200">I've been seeing similar results with our campaigns. Let's connect and share some notes!</p>
                </div>
                <div className="flex items-center mt-1 space-x-2 text-xs text-gray-500 dark:text-gray-400">
                  <button className="font-medium hover:text-gray-700 dark:hover:text-gray-300">Like</button>
                  <button className="font-medium hover:text-gray-700 dark:hover:text-gray-300">Reply</button>
                  <span>1h</span>
                </div>
              </div>
            </div>
            
            {post.comments > 2 && (
              <button className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 font-medium">
                View all {post.comments} comments
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PostCard;