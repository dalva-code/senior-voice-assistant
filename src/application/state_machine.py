from loguru import logger

from src.domain.models import VoiceState


class InvalidStateTransitionError(RuntimeError):
    """Señala una transición no permitida por la máquina de estados."""


class VoiceStateMachine:
    _VALID_TRANSITIONS: dict[VoiceState, frozenset[VoiceState]] = {
        VoiceState.IDLE: frozenset(
            {VoiceState.WAKE_DETECTED, VoiceState.PROMPTING_MESSAGE}
        ),
        VoiceState.WAKE_DETECTED: frozenset(
            {VoiceState.LISTENING, VoiceState.IDLE}
        ),
        VoiceState.LISTENING: frozenset(
            {VoiceState.THINKING, VoiceState.IDLE}
        ),
        VoiceState.THINKING: frozenset(
            {
                VoiceState.SPEAKING,
                VoiceState.PLAYING_MESSAGE,
                VoiceState.PROMPTING_MESSAGE,
                VoiceState.IDLE,
            }
        ),
        VoiceState.SPEAKING: frozenset(
            {
                VoiceState.AWAITING_FOLLOWUP,
                VoiceState.IDLE,
                VoiceState.INTERRUPTED,
            }
        ),
        VoiceState.AWAITING_FOLLOWUP: frozenset(
            {
                VoiceState.LISTENING,
                VoiceState.IDLE,
                VoiceState.PROMPTING_MESSAGE,
            }
        ),
        VoiceState.INTERRUPTED: frozenset(
            {VoiceState.LISTENING, VoiceState.IDLE}
        ),
        VoiceState.PROMPTING_MESSAGE: frozenset(
            {VoiceState.PLAYING_MESSAGE, VoiceState.IDLE}
        ),
        VoiceState.PLAYING_MESSAGE: frozenset(
            {VoiceState.PROMPTING_REPLY, VoiceState.IDLE}
        ),
        VoiceState.PROMPTING_REPLY: frozenset(
            {
                VoiceState.RECORDING_REPLY,
                VoiceState.SENDING_REPLY,
                VoiceState.IDLE,
            }
        ),
        VoiceState.RECORDING_REPLY: frozenset(
            {VoiceState.SENDING_REPLY, VoiceState.IDLE}
        ),
        VoiceState.SENDING_REPLY: frozenset(
            {VoiceState.IDLE}
        ),
    }

    def __init__(self) -> None:
        self._state = VoiceState.IDLE

    @property
    def state(self) -> VoiceState:
        return self._state

    def can_transition_to(self, target: VoiceState) -> bool:
        return target in self._VALID_TRANSITIONS[self._state]

    def transition_to(self, target: VoiceState) -> None:
        previous = self._state
        if not self.can_transition_to(target):
            raise InvalidStateTransitionError(
                f"Transición de voz inválida: {previous.value} -> {target.value}"
            )
        self._state = target
        logger.bind(
            component="fsm",
            previous_state=previous.value,
            current_state=target.value,
        ).info(
            "Transición de estado: {} -> {}",
            previous.value,
            target.value,
        )
