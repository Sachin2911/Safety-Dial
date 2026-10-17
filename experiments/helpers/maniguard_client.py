"""Local OpenPI client that tolerates long first-query JAX compilation."""

import websockets.sync.client
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class StudyClient(WebsocketClientPolicy):
    def _wait_for_server(self):
        # Upstream inference blocks the server's event loop while JAX compiles.
        # A client heartbeat then mistakes compilation for a dead connection.
        # The batch's explicit process timeout still bounds an unresponsive query.
        connection = websockets.sync.client.connect(
            self._uri, compression=None, max_size=None, ping_interval=None, open_timeout=30,
        )
        return connection, msgpack_numpy.unpackb(connection.recv())
