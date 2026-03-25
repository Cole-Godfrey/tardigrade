# Tardigrade + CrewAI Integration Guide

## Overview

This guide shows how to combine CrewAI task orchestration with Tardigrade resilience features.

## What the example demonstrates

- retries around flaky task execution
- checkpointing workflow state between steps
- one additional resilience feature such as a circuit breaker

## Example flow

1. Configure a mock CrewAI task that raises a transient exception on the first run.
2. Wrap the task with Tardigrade retry and checkpoint middleware.
3. Show how the second run restores state from the checkpoint and completes successfully.
4. Document which environment variables or mock values a new user must change before adapting the example.
