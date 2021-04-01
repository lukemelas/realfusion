import { Button, Stack } from '@chakra-ui/react'
import { AiOutlineGithub } from "react-icons/ai"
import { IoIosPaper } from "react-icons/io"

import NextLink from 'next/link'

const links = {
  'paper': "https://arxiv.org/abs/2002.00733",
  'github': "https://github.com/lukemelas"
}

export const LinksRow = () => (
  <Stack direction="row" spacing={4} pt="2rem" pb="2rem">
    <NextLink href={links.paper} passHref={true}>
      <Button leftIcon={<IoIosPaper size="25px" />} colorScheme="teal" variant="outline">
        Paper
      </Button>
    </NextLink>
    <NextLink href={links.github} passHref={true}>
      <Button leftIcon={<AiOutlineGithub size="25px" />} colorScheme="teal" variant="solid">
        GitHub
      </Button>
    </NextLink>
  </Stack >
)

