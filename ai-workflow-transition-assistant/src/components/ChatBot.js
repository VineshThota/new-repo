import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';

const ChatContainer = styled(motion.div)`
  display: flex;
  flex-direction: column;
  height: 70vh;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  overflow: hidden;
`;

const ChatHeader = styled.div`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
  color: white;
  text-align: center;
  font-weight: 600;
  font-size: 1.1rem;
`;

const MessagesContainer = styled.div`
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 15px;
`;

const Message = styled(motion.div)`
  display: flex;
  align-items: flex-start;
  gap: 10px;
  ${props => props.isUser ? 'flex-direction: row-reverse;' : ''}
`;

const MessageBubble = styled.div`
  max-width: 70%;
  padding: 15px 20px;
  border-radius: 20px;
  ${props => props.isUser ? `
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-bottom-right-radius: 5px;
  ` : `
    background: rgba(255, 255, 255, 0.9);
    color: #333;
    border-bottom-left-radius: 5px;
  `}
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
`;

const Avatar = styled.div`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  color: white;
  ${props => props.isUser ? `
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  ` : `
    background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
  `}
`;

const InputContainer = styled.div`
  padding: 20px;
  background: rgba(255, 255, 255, 0.05);
  border-top: 1px solid rgba(255, 255, 255, 0.1);
`;

const InputWrapper = styled.div`
  display: flex;
  gap: 10px;
  align-items: center;
`;

const MessageInput = styled.input`
  flex: 1;
  padding: 15px 20px;
  border: none;
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.9);
  color: #333;
  font-size: 1rem;
  outline: none;
  
  &::placeholder {
    color: #666;
  }
`;

const SendButton = styled(motion.button)`
  background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
  border: none;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const QuickReplies = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 15px;
`;

const QuickReply = styled(motion.button)`
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 20px;
  padding: 8px 15px;
  color: white;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
  }
`;

const TypingIndicator = styled(motion.div)`
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 15px 20px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 20px;
  border-bottom-left-radius: 5px;
  max-width: 70px;
`;

const TypingDot = styled(motion.div)`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #666;
`;

const ChatBot = ({ user, onAddNotification }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: `Hello ${user.name}! I'm your AI Transition Assistant. I'm here to help you navigate the transition to AI-powered workflows. What concerns or questions do you have about this change?`,
      isUser: false,
      timestamp: new Date().toISOString()
    }
  ]);
  
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const quickReplies = [
    "Will AI replace my job?",
    "How do I learn new AI tools?",
    "What if I make mistakes?",
    "How will this affect my team?",
    "What are the benefits for me?",
    "How long will the transition take?"
  ];

  const aiResponses = {
    "Will AI replace my job?": "AI is designed to augment your capabilities, not replace you. Our transition program focuses on upskilling you to work alongside AI, making you more valuable and efficient. Studies show that employees who embrace AI tools often see career advancement and increased job satisfaction.",
    
    "How do I learn new AI tools?": "We provide step-by-step training modules tailored to your role and learning pace. You'll start with basic concepts and gradually progress to advanced features. Our hands-on approach ensures you're comfortable before moving to the next level.",
    
    "What if I make mistakes?": "Making mistakes is part of learning! Our AI systems have built-in safeguards and our training environment is designed for safe experimentation. You'll have mentors and support available whenever you need help.",
    
    "How will this affect my team?": "Your entire team will go through this transition together. We're fostering a collaborative learning environment where team members support each other. Many teams report stronger bonds and improved communication after the transition.",
    
    "What are the benefits for me?": "You'll gain valuable AI skills that are highly sought after in today's job market. You'll spend less time on repetitive tasks and more time on creative, strategic work. Many employees report increased job satisfaction and faster career growth.",
    
    "How long will the transition take?": "The transition is gradual and personalized. Most employees feel comfortable with basic AI tools within 2-4 weeks, and become proficient within 2-3 months. We move at your pace to ensure you're confident at each step."
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (messageText = inputValue) => {
    if (!messageText.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: messageText,
      isUser: true,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    // Simulate AI response delay
    setTimeout(() => {
      const response = generateAIResponse(messageText);
      const aiMessage = {
        id: Date.now() + 1,
        text: response,
        isUser: false,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, aiMessage]);
      setIsTyping(false);
      
      // Add notification for important topics
      if (messageText.toLowerCase().includes('job') || messageText.toLowerCase().includes('replace')) {
        onAddNotification({
          type: 'info',
          message: 'Remember: AI is here to enhance your capabilities, not replace you!'
        });
      }
    }, 1500);
  };

  const generateAIResponse = (userMessage) => {
    const lowerMessage = userMessage.toLowerCase();
    
    // Check for exact quick reply matches
    for (const [question, answer] of Object.entries(aiResponses)) {
      if (userMessage === question) {
        return answer;
      }
    }
    
    // Pattern matching for common concerns
    if (lowerMessage.includes('job') && (lowerMessage.includes('lose') || lowerMessage.includes('replace'))) {
      return aiResponses["Will AI replace my job?"];
    }
    
    if (lowerMessage.includes('learn') || lowerMessage.includes('training')) {
      return aiResponses["How do I learn new AI tools?"];
    }
    
    if (lowerMessage.includes('mistake') || lowerMessage.includes('error') || lowerMessage.includes('wrong')) {
      return aiResponses["What if I make mistakes?"];
    }
    
    if (lowerMessage.includes('team') || lowerMessage.includes('colleague')) {
      return aiResponses["How will this affect my team?"];
    }
    
    if (lowerMessage.includes('benefit') || lowerMessage.includes('advantage')) {
      return aiResponses["What are the benefits for me?"];
    }
    
    if (lowerMessage.includes('time') || lowerMessage.includes('long') || lowerMessage.includes('duration')) {
      return aiResponses["How long will the transition take?"];
    }
    
    // Default responses for other queries
    const defaultResponses = [
      "That's a great question! The key to successful AI adoption is taking it step by step. Would you like me to explain how our training program addresses your specific concerns?",
      "I understand your concern. Many employees have similar questions. Our approach is designed to make this transition as smooth as possible. What specific aspect would you like to know more about?",
      "Thank you for sharing that with me. Change can feel overwhelming, but you're not alone in this journey. Our support system is designed to help you succeed. How can I best assist you today?",
      "That's an important consideration. Our experience shows that employees who engage with the transition process early tend to have the most positive outcomes. What would help you feel more confident about this change?"
    ];
    
    return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <ChatContainer
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <ChatHeader>
        🤖 AI Transition Assistant - Here to Help You Succeed
      </ChatHeader>
      
      <MessagesContainer>
        <AnimatePresence>
          {messages.map((message) => (
            <Message
              key={message.id}
              isUser={message.isUser}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Avatar isUser={message.isUser}>
                {message.isUser ? user.name.charAt(0).toUpperCase() : '🤖'}
              </Avatar>
              <MessageBubble isUser={message.isUser}>
                {message.text}
              </MessageBubble>
            </Message>
          ))}
          
          {isTyping && (
            <Message isUser={false}>
              <Avatar isUser={false}>🤖</Avatar>
              <TypingIndicator
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
              >
                {[0, 1, 2].map((i) => (
                  <TypingDot
                    key={i}
                    animate={{ y: [0, -5, 0] }}
                    transition={{
                      duration: 0.6,
                      repeat: Infinity,
                      delay: i * 0.2
                    }}
                  />
                ))}
              </TypingIndicator>
            </Message>
          )}
        </AnimatePresence>
        
        {messages.length === 1 && (
          <QuickReplies>
            {quickReplies.map((reply, index) => (
              <QuickReply
                key={index}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => handleSendMessage(reply)}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 + index * 0.1 }}
              >
                {reply}
              </QuickReply>
            ))}
          </QuickReplies>
        )}
        
        <div ref={messagesEndRef} />
      </MessagesContainer>
      
      <InputContainer>
        <InputWrapper>
          <MessageInput
            type="text"
            placeholder="Ask me anything about AI workflow transition..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isTyping}
          />
          <SendButton
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => handleSendMessage()}
            disabled={isTyping || !inputValue.trim()}
          >
            ➤
          </SendButton>
        </InputWrapper>
      </InputContainer>
    </ChatContainer>
  );
};

export default ChatBot;