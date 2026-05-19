import rclpy, asyncio, json, threading
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import websockets

latest_sectors = [0.0] * 36

class VizNode(Node):
    def __init__(self):
        super().__init__('viz_node')
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

    def scan_callback(self, msg):
        global latest_sectors
        ranges = list(msg.ranges)
        sectors = []
        for i in range(36):
            chunk = ranges[i*10:(i*10)+10]
            valid = [r for r in chunk if msg.range_min < r < msg.range_max]
            min_r = min(valid) if valid else 3.5
            sectors.append(round(max(0.0, 1.0 - min_r / msg.range_max), 3))
        latest_sectors = sectors[:1] + sectors[1:][::-1]

async def ws_handler(websocket):
    while True:
        await websocket.send(json.dumps(latest_sectors))
        await asyncio.sleep(0.1)

async def main():
    # Fixed: use async with instead of asyncio.run(websockets.serve(...))
    async with websockets.serve(ws_handler, 'localhost', 8765):
        print('Websocket server running on ws://localhost:8765')
        await asyncio.Future()  # keeps server alive forever

def ros_thread():
    rclpy.init()
    rclpy.spin(VizNode())

threading.Thread(target=ros_thread, daemon=True).start()
asyncio.run(main())