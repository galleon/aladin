"""Usage statistics API for the admin dashboard."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..database import get_db
from ..models import User, Message, Conversation, TranslationJob
from ..services.auth import get_current_active_user

router = APIRouter(prefix="/stats", tags=["Statistics"])


@router.get("/usage")
async def get_usage_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Return prompts count and token usage for the current user.

    - **prompts_count**: Number of user messages (prompts) in conversations.
    - **input_tokens**: Total input tokens from chat (assistant messages) and translation jobs.
    - **output_tokens**: Total output tokens from chat and translation jobs.
    - **total_tokens**: input_tokens + output_tokens.
    """
    user_id = current_user.id
    try:
        # Prompts: count of user messages in conversations owned by this user
        prompts_count = (
            db.query(func.count(Message.id))
            .join(Conversation, Message.conversation_id == Conversation.id)
            .filter(Conversation.user_id == user_id, Message.role == "user")
            .scalar()
            or 0
        )

        # Tokens from assistant messages (chat)
        msg_tokens = (
            db.query(
                func.coalesce(func.sum(Message.input_tokens), 0).label("input"),
                func.coalesce(func.sum(Message.output_tokens), 0).label("output"),
            )
            .join(Conversation, Message.conversation_id == Conversation.id)
            .filter(Conversation.user_id == user_id, Message.role == "assistant")
            .one()
        )
        chat_input = int(msg_tokens.input or 0)
        chat_output = int(msg_tokens.output or 0)

        # Tokens from translation jobs
        job_tokens = (
            db.query(
                func.coalesce(func.sum(TranslationJob.input_tokens), 0).label("input"),
                func.coalesce(func.sum(TranslationJob.output_tokens), 0).label("output"),
            )
            .filter(TranslationJob.user_id == user_id)
            .one()
        )
        job_input = int(job_tokens.input or 0)
        job_output = int(job_tokens.output or 0)

        input_tokens = chat_input + job_input
        output_tokens = chat_output + job_output
        total_tokens = input_tokens + output_tokens

        return {
            "prompts_count": prompts_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
    except Exception:
        return {
            "prompts_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
