import React, { useState, useEffect } from 'react';
import { useUser } from '../contexts/UserContext';
import { Search, User, Send, Phone, Video, MoreVertical, ChevronLeft } from 'lucide-react';
import { format } from 'date-fns';

interface Contact {
  id: string;
  name: string;
  profileImage?: string;
  lastMessage?: string;
  lastMessageTime?: number;
  online: boolean;
  unread: number;
}

interface Message {
  id: string;
  senderId: string;
  receiverId: string;
  text: string;
  timestamp: number;
  read: boolean;
}

const MessagesPage: React.FC = () => {
  const { user } = useUser();
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [selectedContact, setSelectedContact] = useState<Contact | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [showMobileConversation, setShowMobileConversation] = useState(false);
  
  // Mock data initialization
  useEffect(() => {
    // Simulate API call to fetch contacts
    const mockContacts: Contact[] = [
      {
        id: 'contact1',
        name: 'Sarah Johnson',
        profileImage: 'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg',
        lastMessage: 'Let me know if you need any help with the campaign setup.',
        lastMessageTime: Date.now() - 1800000, // 30 minutes ago
        online: true,
        unread: 2
      },
      {
        id: 'contact2',
        name: 'Miguel Lopez',
        profileImage: 'https://images.pexels.com/photos/2379005/pexels-photo-2379005.jpeg',
        lastMessage: 'The stats for yesterday\'s campaign look great!',
        lastMessageTime: Date.now() - 7200000, // 2 hours ago
        online: false,
        unread: 0
      },
      {
        id: 'contact3',
        name: 'Priya Patel',
        profileImage: 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg',
        lastMessage: 'Can we discuss the new affiliate program tomorrow?',
        lastMessageTime: Date.now() - 86400000, // 1 day ago
        online: true,
        unread: 0
      },
      {
        id: 'contact4',
        name: 'David Wilson',
        profileImage: 'https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg',
        lastMessage: 'Thanks for the introduction to the new platform.',
        lastMessageTime: Date.now() - 172800000, // 2 days ago
        online: false,
        unread: 0
      }
    ];
    
    setContacts(mockContacts);
  }, []);
  
  // Load messages when selecting a contact
  useEffect(() => {
    if (selectedContact) {
      // Simulate API call to fetch messages for selected contact
      const mockMessages: Message[] = [
        {
          id: 'msg1',
          senderId: selectedContact.id,
          receiverId: user?.id || '',
          text: 'Hey, how are things going with the new campaign?',
          timestamp: Date.now() - 86400000 - 3600000, // 1 day and 1 hour ago
          read: true
        },
        {
          id: 'msg2',
          senderId: user?.id || '',
          receiverId: selectedContact.id,
          text: 'It\'s coming along well! We\'re seeing good initial results.',
          timestamp: Date.now() - 86400000 - 3540000, // 1 day and 59 minutes ago
          read: true
        },
        {
          id: 'msg3',
          senderId: selectedContact.id,
          receiverId: user?.id || '',
          text: 'That\'s great to hear! What kind of CTR are you seeing?',
          timestamp: Date.now() - 86400000 - 3480000, // 1 day and 58 minutes ago
          read: true
        },
        {
          id: 'msg4',
          senderId: user?.id || '',
          receiverId: selectedContact.id,
          text: 'Around 3.2% so far, which is about 0.8% higher than our previous benchmark.',
          timestamp: Date.now() - 86400000 - 3420000, // 1 day and 57 minutes ago
          read: true
        },
        {
          id: 'msg5',
          senderId: selectedContact.id,
          receiverId: user?.id || '',
          text: 'Let me know if you need any help with the campaign setup.',
          timestamp: Date.now() - 1800000, // 30 minutes ago
          read: false
        }
      ];
      
      setMessages(mockMessages);
      
      // Mark unread messages as read when conversation is opened
      if (selectedContact.unread > 0) {
        setContacts(prevContacts => 
          prevContacts.map(contact => 
            contact.id === selectedContact.id ? { ...contact, unread: 0 } : contact
          )
        );
      }
      
      // Set mobile view to show conversation
      setShowMobileConversation(true);
    }
  }, [selectedContact, user?.id]);
  
  // Send a new message
  const handleSendMessage = () => {
    if (newMessage.trim() === '' || !selectedContact) return;
    
    const newMsg: Message = {
      id: `msg${Date.now()}`,
      senderId: user?.id || '',
      receiverId: selectedContact.id,
      text: newMessage,
      timestamp: Date.now(),
      read: false
    };
    
    setMessages([...messages, newMsg]);
    setNewMessage('');
    
    // Update last message in contacts list
    setContacts(prevContacts => 
      prevContacts.map(contact => 
        contact.id === selectedContact.id 
          ? { 
              ...contact, 
              lastMessage: newMessage, 
              lastMessageTime: Date.now() 
            } 
          : contact
      )
    );
  };
  
  // Format timestamp for messages
  const formatMessageTime = (timestamp: number) => {
    const messageDate = new Date(timestamp);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    
    if (messageDate.toDateString() === today.toDateString()) {
      return format(messageDate, 'h:mm a');
    } else if (messageDate.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    } else {
      return format(messageDate, 'MMM d');
    }
  };
  
  // Filter contacts based on search
  const filteredContacts = contacts.filter(contact => 
    contact.name.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  return (
    <div className="pb-16 lg:pb-0 h-[calc(100vh-10rem)]">
      <div className="h-full bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <div className="flex h-full">
          {/* Contacts sidebar - hidden on mobile when conversation is open */}
          <div className={`${showMobileConversation ? 'hidden md:block' : 'block'} w-full md:w-80 lg:w-96 border-r border-gray-200 dark:border-gray-700 flex flex-col`}>
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Messages</h2>
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search messages"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-white"
                />
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
              </div>
            </div>
            
            {/* Contacts list */}
            <div className="flex-1 overflow-y-auto">
              {filteredContacts.length > 0 ? (
                <ul className="divide-y divide-gray-200 dark:divide-gray-700">
                  {filteredContacts.map((contact) => (
                    <li 
                      key={contact.id}
                      onClick={() => setSelectedContact(contact)}
                      className={`cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 ${
                        selectedContact?.id === contact.id ? 'bg-primary-50 dark:bg-primary-900/20' : ''
                      }`}
                    >
                      <div className="flex items-center px-4 py-3 relative">
                        <div className="relative flex-shrink-0">
                          <div className="h-12 w-12 rounded-full bg-primary-100 overflow-hidden">
                            {contact.profileImage ? (
                              <img src={contact.profileImage} alt={contact.name} className="h-full w-full object-cover" />
                            ) : (
                              <div className="h-full w-full flex items-center justify-center">
                                <User className="h-6 w-6 text-primary-600" />
                              </div>
                            )}
                          </div>
                          {contact.online && (
                            <span className="absolute bottom-0 right-0 block h-3 w-3 rounded-full bg-success-500 ring-2 ring-white dark:ring-gray-800"></span>
                          )}
                        </div>
                        <div className="ml-3 flex-1 min-w-0">
                          <div className="flex items-center justify-between">
                            <p className={`text-sm font-medium truncate ${
                              contact.unread > 0 
                                ? 'text-gray-900 dark:text-white' 
                                : 'text-gray-700 dark:text-gray-300'
                            }`}>
                              {contact.name}
                            </p>
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                              {contact.lastMessageTime && formatMessageTime(contact.lastMessageTime)}
                            </p>
                          </div>
                          <div className="flex items-center justify-between">
                            <p className={`text-sm truncate ${
                              contact.unread > 0 
                                ? 'text-gray-900 dark:text-white font-medium' 
                                : 'text-gray-500 dark:text-gray-400'
                            }`}>
                              {contact.lastMessage}
                            </p>
                            {contact.unread > 0 && (
                              <span className="inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white bg-primary-600 rounded-full">
                                {contact.unread}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="py-6 text-center">
                  <p className="text-gray-500 dark:text-gray-400">No contacts found</p>
                </div>
              )}
            </div>
          </div>
          
          {/* Conversation area - shows when contact is selected */}
          <div className={`${showMobileConversation ? 'block' : 'hidden md:block'} flex-1 flex flex-col h-full`}>
            {selectedContact ? (
              <>
                {/* Conversation header */}
                <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
                  <div className="flex items-center">
                    <button 
                      className="md:hidden mr-2 text-gray-500 dark:text-gray-400"
                      onClick={() => setShowMobileConversation(false)}
                    >
                      <ChevronLeft className="h-6 w-6" />
                    </button>
                    <div className="relative flex-shrink-0">
                      <div className="h-10 w-10 rounded-full bg-primary-100 overflow-hidden">
                        {selectedContact.profileImage ? (
                          <img src={selectedContact.profileImage} alt={selectedContact.name} className="h-full w-full object-cover" />
                        ) : (
                          <div className="h-full w-full flex items-center justify-center">
                            <User className="h-5 w-5 text-primary-600" />
                          </div>
                        )}
                      </div>
                      {selectedContact.online && (
                        <span className="absolute bottom-0 right-0 block h-2.5 w-2.5 rounded-full bg-success-500 ring-2 ring-white dark:ring-gray-800"></span>
                      )}
                    </div>
                    <div className="ml-3">
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        {selectedContact.name}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {selectedContact.online ? 'Online' : 'Offline'}
                      </p>
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <button className="p-1.5 rounded-full text-gray-500 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700">
                      <Phone className="h-5 w-5" />
                    </button>
                    <button className="p-1.5 rounded-full text-gray-500 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700">
                      <Video className="h-5 w-5" />
                    </button>
                    <button className="p-1.5 rounded-full text-gray-500 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700">
                      <MoreVertical className="h-5 w-5" />
                    </button>
                  </div>
                </div>
                
                {/* Messages */}
                <div className="flex-1 p-4 overflow-y-auto">
                  <div className="space-y-4">
                    {messages.map((message) => {
                      const isOwnMessage = message.senderId === user?.id;
                      
                      return (
                        <div 
                          key={message.id} 
                          className={`flex ${isOwnMessage ? 'justify-end' : 'justify-start'}`}
                        >
                          <div 
                            className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                              isOwnMessage 
                                ? 'bg-primary-500 text-white rounded-br-none' 
                                : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white rounded-bl-none'
                            }`}
                          >
                            <p className="text-sm">{message.text}</p>
                            <p 
                              className={`text-xs mt-1 text-right ${
                                isOwnMessage ? 'text-primary-100' : 'text-gray-500 dark:text-gray-400'
                              }`}
                            >
                              {formatMessageTime(message.timestamp)}
                            </p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
                
                {/* Message input */}
                <div className="p-4 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex items-center">
                    <input
                      type="text"
                      placeholder="Type a message..."
                      value={newMessage}
                      onChange={(e) => setNewMessage(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                      className="flex-1 py-2 px-4 border border-gray-300 dark:border-gray-600 rounded-l-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-white"
                    />
                    <button
                      onClick={handleSendMessage}
                      disabled={newMessage.trim() === ''}
                      className="py-2 px-4 bg-primary-500 text-white rounded-r-md hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <Send className="h-5 w-5" />
                    </button>
                  </div>
                </div>
              </>
            ) : (
              // No conversation selected
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center px-6">
                  <div className="inline-flex items-center justify-center h-16 w-16 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 mb-4">
                    <MessageSquare className="h-8 w-8" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">Your Messages</h3>
                  <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                    Select a conversation or start a new one
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Import the MessageSquare icon at the top level
import { MessageSquare } from 'lucide-react';

export default MessagesPage;