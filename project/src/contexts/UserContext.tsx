import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  id: string;
  name: string;
  email: string;
  userType: 'Affiliate' | 'Traffic Manager' | 'Agency' | 'Company' | 'Influencer';
  profileImage?: string;
  bio?: string;
  connections?: number;
}

interface UserContextType {
  user: User | null;
  isAuthenticated: boolean;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (userData: Partial<User> & { password: string }) => Promise<void>;
  logout: () => void;
  updateUser: (userData: Partial<User>) => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export const useUser = () => {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};

interface UserProviderProps {
  children: ReactNode;
}

export const UserProvider: React.FC<UserProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  // Load user from localStorage on initial mount
  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setUser(parsedUser);
        setIsAuthenticated(true);
      } catch (error) {
        console.error('Failed to parse stored user:', error);
        localStorage.removeItem('user');
      }
    }
    setLoading(false);
  }, []);

  // Mock login function
  const login = async (email: string, password: string) => {
    setLoading(true);
    try {
      // In a real app, this would be an API call to your Django backend
      // Simulating API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock user for demo purposes
      const mockUser: User = {
        id: 'user123',
        name: 'Jane Smith',
        email: email,
        userType: 'Affiliate',
        profileImage: 'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg',
        bio: 'Performance marketing specialist with 5+ years of experience in affiliate marketing and paid social.',
        connections: 142
      };
      
      setUser(mockUser);
      setIsAuthenticated(true);
      
      // Save to localStorage
      localStorage.setItem('user', JSON.stringify(mockUser));
    } catch (error) {
      console.error('Login failed:', error);
      throw new Error('Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  // Mock register function
  const register = async (userData: Partial<User> & { password: string }) => {
    setLoading(true);
    try {
      // In a real app, this would be an API call to your Django backend
      // Simulating API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Create a new user
      const newUser: User = {
        id: `user${Date.now()}`,
        name: userData.name || 'New User',
        email: userData.email || '',
        userType: userData.userType || 'Affiliate',
        bio: userData.bio,
        connections: 0
      };
      
      setUser(newUser);
      setIsAuthenticated(true);
      
      // Save to localStorage
      localStorage.setItem('user', JSON.stringify(newUser));
    } catch (error) {
      console.error('Registration failed:', error);
      throw new Error('Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Logout function
  const logout = () => {
    setUser(null);
    setIsAuthenticated(false);
    localStorage.removeItem('user');
  };

  // Update user function
  const updateUser = (userData: Partial<User>) => {
    if (user) {
      const updatedUser = { ...user, ...userData };
      setUser(updatedUser);
      localStorage.setItem('user', JSON.stringify(updatedUser));
    }
  };

  return (
    <UserContext.Provider value={{ 
      user, 
      isAuthenticated, 
      loading, 
      login, 
      register, 
      logout, 
      updateUser 
    }}>
      {children}
    </UserContext.Provider>
  );
};