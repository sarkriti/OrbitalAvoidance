"""
Singleton propagator to ensure only one Orekit JVM instance exists.
"""
import logging
from typing import Optional
from orekit_physics import OrekitOrbitPropagator

logger = logging.getLogger(__name__)

class SingletonPropagator:
    """Singleton class to manage a single Orekit propagator instance."""
    
    _instance: Optional[OrekitOrbitPropagator] = None
    _initialized: bool = False
    
    @classmethod
    def get_instance(cls) -> OrekitOrbitPropagator:
        """Get the singleton propagator instance."""
        if cls._instance is None or not cls._initialized:
            logger.info("🔄 Creating singleton Orekit propagator...")
            try:
                cls._instance = OrekitOrbitPropagator()
                cls._initialized = True
                logger.info("✅ Singleton Orekit propagator created successfully")
            except Exception as e:
                logger.error(f"❌ Failed to create singleton propagator: {e}")
                raise
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing/cleanup)."""
        cls._instance = None
        cls._initialized = False
        logger.info("🔄 Singleton propagator reset")
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the singleton is initialized."""
        return cls._initialized and cls._instance is not None


def get_singleton_propagator() -> OrekitOrbitPropagator:
    """Convenience function to get the singleton propagator."""
    return SingletonPropagator.get_instance()