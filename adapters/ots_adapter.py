# adapters/ots_adapter.py

from common.structured_log import slog

from core.types import TimeStampedPacket
from adapters import BaseBlockingAdapter
from natnet import NatNetClient

class OTSAdapter(BaseBlockingAdapter):
    def __init__(self, loop, out_queue, config, source_name: str = "ots"):
        super().__init__(loop, out_queue, source_name)
        # NatNet Client Initialization
        self.client = NatNetClient.NatNetClient()
        self.client.initialize()
        
        # Set up the client to receive specific data from the Motive application
        self.client.setDataCallback(self.data_callback)
    def start(self):
        slog(
            component=self.__class__.__name__,
            event="start",
            source=self.source_name,
        )
        # Start the NatNet client in its own thread
        self.client.run()
        
    def stop(self):
        slog(
            component=self.__class__.__name__,
            event="stop",
            source=self.source_name,
        )
        self.client.stop()
        super().stop()
    def data_callback(self, data):
        """
        Callback function to handle the data received from NatNet.
        This will be called whenever new data is available.
        """
        # Assume 'data' is a dictionary or object that contains the required data
        # Process the data and create a TimeStampedPacket
        
        timestamp = data.timestamp  # Example timestamp from the NatNet data
        packet_data = data.payload  # Example payload (data from the NatNet client)

        pkt = TimeStampedPacket(
            t_host=timestamp,  # Using the timestamp from the data
            source=self.source_name,
            payload={"stream": "ots", "data": packet_data},
        )

        self.loop.call_soon_threadsafe(self.out_queue.put_nowait, pkt)
    def _read_from_sdk_blocking(self):
        # TODO: 从 OTS/NatNet 获取最新 frame
        # return self.client.get_frame()
        return None
