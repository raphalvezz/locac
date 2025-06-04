import React, { useState, useRef } from 'react';
import { useUser } from '../../contexts/UserContext';
import { Image, Video, Link2, X, User } from 'lucide-react';
import { Post } from '../../types/post';

interface PostCreationCardProps {
  onPostCreated: (post: Post) => void;
}

const PostCreationCard: React.FC<PostCreationCardProps> = ({ onPostCreated }) => {
  const { user } = useUser();
  const [content, setContent] = useState('');
  const [mediaPreview, setMediaPreview] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<'image' | 'video' | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Handle media file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Check file type
    if (file.type.startsWith('image/')) {
      setMediaType('image');
    } else if (file.type.startsWith('video/')) {
      setMediaType('video');
    } else {
      alert('Unsupported file type');
      return;
    }
    
    // Create preview URL
    const fileURL = URL.createObjectURL(file);
    setMediaPreview(fileURL);
  };
  
  // Clear media preview
  const clearMedia = () => {
    if (mediaPreview) {
      URL.revokeObjectURL(mediaPreview);
    }
    setMediaPreview(null);
    setMediaType(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  // Create new post
  const createPost = () => {
    if (!user) return;
    if (!content.trim() && !mediaPreview) return;
    
    const newPost: Post = {
      id: `post${Date.now()}`,
      author: {
        id: user.id,
        name: user.name,
        userType: user.userType,
        profileImage: user.profileImage
      },
      content: content.trim(),
      timestamp: Date.now(),
      likes: 0,
      comments: 0,
      shares: 0
    };
    
    // Add media if present
    if (mediaPreview && mediaType) {
      newPost.media = [
        {
          type: mediaType,
          url: mediaPreview,
          alt: 'Post media'
        }
      ];
    }
    
    // Call the callback with the new post
    onPostCreated(newPost);
    
    // Reset form
    setContent('');
    clearMedia();
  };
  
  // Handle text area auto-resize
  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContent(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${e.target.scrollHeight}px`;
  };
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
      <div className="p-4">
        {/* User info and input */}
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <div className="h-10 w-10 rounded-full bg-primary-100 overflow-hidden">
              {user?.profileImage ? (
                <img src={user.profileImage} alt={user.name} className="h-full w-full object-cover" />
              ) : (
                <div className="h-full w-full flex items-center justify-center">
                  <User className="h-5 w-5 text-primary-600" />
                </div>
              )}
            </div>
          </div>
          <div className="ml-3 flex-1">
            <textarea
              placeholder="What's on your mind?"
              value={content}
              onChange={handleTextareaChange}
              className="w-full border-none focus:ring-0 resize-none text-gray-900 dark:text-white bg-transparent"
              style={{ minHeight: '60px' }}
            />
          </div>
        </div>
        
        {/* Media preview */}
        {mediaPreview && (
          <div className="mt-3 relative">
            <button
              onClick={clearMedia}
              className="absolute top-2 right-2 bg-gray-800 bg-opacity-70 text-white rounded-full p-1"
            >
              <X className="h-4 w-4" />
            </button>
            {mediaType === 'image' && (
              <img
                src={mediaPreview}
                alt="Post preview"
                className="w-full h-auto rounded-lg max-h-96 object-contain"
              />
            )}
            {mediaType === 'video' && (
              <video
                src={mediaPreview}
                controls
                className="w-full h-auto rounded-lg max-h-96"
              />
            )}
          </div>
        )}
      </div>
      
      {/* Attachment options */}
      <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 flex flex-wrap items-center justify-between">
        <div className="flex space-x-2">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 px-2 py-1 rounded-md text-sm"
          >
            <Image className="h-5 w-5 mr-1" />
            <span className="hidden sm:inline">Photo</span>
          </button>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*,video/*"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 px-2 py-1 rounded-md text-sm"
          >
            <Video className="h-5 w-5 mr-1" />
            <span className="hidden sm:inline">Video</span>
          </button>
          <button className="flex items-center text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 px-2 py-1 rounded-md text-sm">
            <Link2 className="h-5 w-5 mr-1" />
            <span className="hidden sm:inline">Link</span>
          </button>
        </div>
        <button
          onClick={createPost}
          disabled={!content.trim() && !mediaPreview}
          className="px-4 py-1.5 bg-primary-600 text-white rounded-md text-sm font-medium hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Post
        </button>
      </div>
    </div>
  );
};

export default PostCreationCard;