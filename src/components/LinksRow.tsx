import { ArrowForwardIcon } from '@chakra-ui/icons'
import { Link as ChakraLink, Button, Stack } from '@chakra-ui/react'

import { Container } from './Container'

export const LinksRow = () => (
  <Stack direction="row" spacing={4} pt="5vh" pb="5vh">
    <Button leftIcon={<ArrowForwardIcon />} colorScheme="teal" variant="solid">
      Email
  </Button>
    <Button rightIcon={<ArrowForwardIcon />} colorScheme="teal" variant="outline">
      Call us
  </Button>
  </Stack>
)

