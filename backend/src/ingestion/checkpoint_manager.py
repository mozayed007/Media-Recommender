"""Checkpoint manager for ingestion pipeline.

Handles saving and loading checkpoint state for resumable data ingestion.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import shutil

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints for data ingestion processes.
    
    Provides atomic writes, versioning, and validation for checkpoint data.
    """
    
    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.version = "1.0"
    
    def save_checkpoint(
        self, 
        source: str, 
        state: Dict[str, Any],
        backup_previous: bool = True
    ) -> Path:
        """Save checkpoint state to file.
        
        Args:
            source: Source identifier (e.g., 'tmdb', 'mal', 'mangadex')
            state: Checkpoint state dictionary
            backup_previous: Whether to backup existing checkpoint
            
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_file = self.checkpoint_dir / f"{source}_checkpoint.json"
        
        # Backup existing checkpoint
        if backup_previous and checkpoint_file.exists():
            backup_path = self._create_backup(checkpoint_file)
            self.logger.debug(f"Created checkpoint backup: {backup_path}")
        
        # Prepare checkpoint data
        checkpoint_data = {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "version": self.version,
            "state": state,
        }
        
        # Atomic write
        temp_file = checkpoint_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            # Rename for atomicity
            temp_file.replace(checkpoint_file)
            
            self.logger.info(f"Saved checkpoint for {source}: {checkpoint_file}")
            return checkpoint_file
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    def load_checkpoint(self, source: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint state from file.
        
        Args:
            source: Source identifier
            
        Returns:
            Checkpoint state dictionary or None if not found
        """
        checkpoint_file = self.checkpoint_dir / f"{source}_checkpoint.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
            
            # Validate checkpoint
            if not self._validate_checkpoint(checkpoint_data):
                self.logger.warning(f"Invalid checkpoint for {source}")
                return None
            
            self.logger.info(f"Loaded checkpoint for {source}")
            return checkpoint_data.get("state")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Corrupted checkpoint file: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self, source: str) -> bool:
        """Clear checkpoint for a source.
        
        Args:
            source: Source identifier
            
        Returns:
            True if checkpoint was cleared, False otherwise
        """
        checkpoint_file = self.checkpoint_dir / f"{source}_checkpoint.json"
        
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                self.logger.info(f"Cleared checkpoint for {source}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to clear checkpoint: {e}")
                return False
        
        return False
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints.
        
        Returns:
            List of source identifiers with checkpoints
        """
        checkpoints = []
        
        for file_path in self.checkpoint_dir.glob("*_checkpoint.json"):
            source = file_path.stem.replace("_checkpoint", "")
            checkpoints.append(source)
        
        return sorted(checkpoints)
    
    def get_checkpoint_age(self, source: str) -> Optional[timedelta]:
        """Get age of checkpoint.
        
        Args:
            source: Source identifier
            
        Returns:
            Timedelta since checkpoint was saved, or None if not found
        """
        checkpoint_file = self.checkpoint_dir / f"{source}_checkpoint.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
            
            timestamp_str = checkpoint_data.get("timestamp")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                return datetime.now() - timestamp
                
        except Exception as e:
            self.logger.warning(f"Failed to get checkpoint age: {e}")
        
        return None
    
    def is_checkpoint_valid(
        self, 
        source: str, 
        max_age_hours: Optional[int] = None
    ) -> bool:
        """Check if checkpoint exists and is valid.
        
        Args:
            source: Source identifier
            max_age_hours: Maximum age in hours for checkpoint to be valid
            
        Returns:
            True if checkpoint is valid, False otherwise
        """
        checkpoint_file = self.checkpoint_dir / f"{source}_checkpoint.json"
        
        if not checkpoint_file.exists():
            return False
        
        # Check age if specified
        if max_age_hours is not None:
            age = self.get_checkpoint_age(source)
            if age is None or age > timedelta(hours=max_age_hours):
                self.logger.info(f"Checkpoint for {source} is too old")
                return False
        
        # Validate structure
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
            return self._validate_checkpoint(checkpoint_data)
        except Exception:
            return False
    
    def _validate_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Validate checkpoint data structure.
        
        Args:
            checkpoint_data: Checkpoint dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["source", "timestamp", "version", "state"]
        
        for field in required_fields:
            if field not in checkpoint_data:
                self.logger.warning(f"Checkpoint missing field: {field}")
                return False
        
        return True
    
    def _create_backup(self, checkpoint_file: Path) -> Path:
        """Create backup of existing checkpoint.
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{checkpoint_file.stem}_backup_{timestamp}.json"
        backup_path = self.checkpoint_dir / backup_name
        
        shutil.copy2(checkpoint_file, backup_path)
        
        # Clean up old backups (keep last 5)
        self._cleanup_old_backups(checkpoint_file.stem)
        
        return backup_path
    
    def _cleanup_old_backups(self, source: str, keep: int = 5):
        """Clean up old checkpoint backups.
        
        Args:
            source: Source identifier
            keep: Number of backups to keep
        """
        pattern = f"{source}_checkpoint_backup_*.json"
        backups = sorted(self.checkpoint_dir.glob(pattern))
        
        if len(backups) > keep:
            for old_backup in backups[:-keep]:
                try:
                    old_backup.unlink()
                    self.logger.debug(f"Removed old backup: {old_backup}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove old backup: {e}")
    
    def get_checkpoint_info(self, source: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint metadata without loading full state.
        
        Args:
            source: Source identifier
            
        Returns:
            Checkpoint metadata or None
        """
        checkpoint_file = self.checkpoint_dir / f"{source}_checkpoint.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
            
            return {
                "source": checkpoint_data.get("source"),
                "timestamp": checkpoint_data.get("timestamp"),
                "version": checkpoint_data.get("version"),
                "age_hours": self.get_checkpoint_age(source).total_seconds() / 3600 if self.get_checkpoint_age(source) else None,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get checkpoint info: {e}")
            return None
    
    def create_initial_checkpoint(
        self, 
        source: str, 
        offset: int = 0,
        extra: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Create initial checkpoint for a new ingestion.
        
        Args:
            source: Source identifier
            offset: Starting offset
            extra: Additional checkpoint data
            
        Returns:
            Path to checkpoint file
        """
        state = {
            "offset": offset,
            "total_fetched": 0,
            "errors": 0,
            "last_id": None,
        }
        
        if extra:
            state.update(extra)
        
        return self.save_checkpoint(source, state)
