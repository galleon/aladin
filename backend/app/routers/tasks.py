"""Tasks router for managing background jobs."""
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Task as TaskModel
from app.schemas import TaskResponse
from app.logger import logger
from app.config import settings
from typing import List
import json
import redis.asyncio as redis
import asyncio

router = APIRouter()

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warn("Failed to send WebSocket message", error=str(e))
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()


@router.get("/", response_model=List[TaskResponse])
async def get_tasks(
    status: str | None = None,
    task_type: str | None = None,
    tenant_id: int | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Get all tasks with optional filters."""
    try:
        query = db.query(TaskModel)

        if status:
            query = query.filter(TaskModel.status == status)
        if task_type:
            query = query.filter(TaskModel.task_type == task_type)
        if tenant_id:
            query = query.filter(TaskModel.tenant_id == tenant_id)

        tasks = query.order_by(TaskModel.created_at.desc()).limit(limit).all()
        return tasks
    except Exception as e:
        logger.error("Failed to get tasks", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch tasks",
        )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, db: Session = Depends(get_db)):
    """Get a specific task by task_id."""
    try:
        task = db.query(TaskModel).filter(TaskModel.task_id == task_id).first()
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found",
            )
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get task", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch task",
        )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time task updates."""
    await manager.connect(websocket)
    r = None
    pubsub = None
    listener_task = None

    try:
        # Connect to Redis using async client
        try:
            r = await redis.from_url(settings.redis_url, decode_responses=True)
            pubsub = r.pubsub()
            await pubsub.subscribe("task_updates")
            logger.info("WebSocket client connected and subscribed to Redis")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e), exc_info=True)
            await websocket.close(code=1011, reason="Redis connection failed")
            return

        # Start listening for Redis messages
        async def listen_redis():
            try:
                while True:
                    try:
                        # Use get_message with timeout
                        message = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True),
                            timeout=1.0
                        )
                        if message and message.get("type") == "message":
                            try:
                                data = json.loads(message["data"])
                                await websocket.send_json(data)
                            except Exception as e:
                                logger.warn("Error sending WebSocket message", error=str(e))
                                # If send fails, connection might be closed
                                return
                    except asyncio.TimeoutError:
                        # Timeout is expected, continue polling
                        continue
                    except asyncio.CancelledError:
                        logger.info("Redis listener cancelled")
                        return
                    except Exception as e:
                        logger.warn("Error in Redis listener", error=str(e))
                        await asyncio.sleep(1)
            except Exception as e:
                logger.warn("Redis listener error", error=str(e))

        # Start Redis listener task
        listener_task = asyncio.create_task(listen_redis())

        # Keep connection alive - receive messages or wait
        try:
            while True:
                try:
                    # Wait for either a message from client or timeout
                    # Use a longer timeout to reduce ping frequency
                    try:
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                        # Echo back pong if client sends ping
                        if data == "ping" or (isinstance(data, str) and data.startswith('{"type":"ping"')):
                            try:
                                await websocket.send_json({"type": "pong"})
                            except Exception:
                                break
                    except asyncio.TimeoutError:
                        # Send periodic ping to keep connection alive
                        try:
                            await websocket.send_json({"type": "ping"})
                        except Exception as e:
                            logger.debug("WebSocket ping failed, connection closed", error=str(e))
                            break
                    except WebSocketDisconnect:
                        logger.info("WebSocket disconnected by client")
                        break
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected by client")
                    break
                except Exception as e:
                    logger.warn("WebSocket receive error", error=str(e))
                    break
        finally:
            if listener_task and not listener_task.done():
                listener_task.cancel()
                try:
                    await listener_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error("WebSocket error", error=str(e), exc_info=True)
    finally:
        if pubsub:
            try:
                await pubsub.unsubscribe("task_updates")
                await pubsub.close()
            except Exception as e:
                logger.debug("Error closing pubsub", error=str(e))
        if r:
            try:
                await r.close()
            except Exception as e:
                logger.debug("Error closing Redis connection", error=str(e))
        try:
            manager.disconnect(websocket)
        except Exception:
            pass
        logger.info("WebSocket cleanup complete")

