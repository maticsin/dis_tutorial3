#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pyttsx3

class TTSNode(Node):
    def __init__(self):  # Corrected the method name here
        super().__init__('tts_node')  # Corrected to __init__
        self.subscription = self.create_subscription(
            String,
            '/tts',
            self.tts_callback,
            10
        )
        # Initialize pyttsx3 for text-to-speech
        self.engine = pyttsx3.init()
        
        self.engine.setProperty('rate', 80)  # npr. 100 za počasnejše branje


        self.get_logger().info("TTS Node started. Waiting for messages on topic '/tts'...")

    def tts_callback(self, msg):
        text_to_speak = msg.data
        self.get_logger().info(f"Speaking: {text_to_speak}")
        # Text-to-speech
        self.engine.say(text_to_speak)
        self.engine.runAndWait()

def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
