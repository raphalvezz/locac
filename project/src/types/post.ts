export interface Post {
  id: string;
  author: {
    id: string;
    name: string;
    userType: string;
    profileImage?: string;
  };
  content: string;
  timestamp: number;
  likes: number;
  comments: number;
  shares: number;
  media?: Media[];
}

export interface Media {
  type: 'image' | 'video';
  url: string;
  alt?: string;
  poster?: string; // For video thumbnails
}